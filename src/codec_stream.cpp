// Streaming NanoCodec decode (see codec_stream.hpp). Adapts NVIDIA's
// NeMo-Speech.cpp state-carrying scheme (src/tts/nanocodec/model.cpp,
// Apache-2.0) to this repo's im2col-based causal conv and the fused grouped
// causal transposed conv, preserving the offline numerics exactly:
//
//   offline codec.cpp                      streaming (here)
//   -------------------------------------------------------------------
//   ggml_pad_ext(left = (k-1)*dil, 0)  ->  cache INPUT [left, C_in]
//                                         concat(cache, x) before im2col
//   (nothing)                          ->  tail OUTPUT = last `left` frames
//                                         of the conv INPUT
//   tconv "x[t-1]" half shifted s,      ->  pre-add PREVIOUS chunk's spill
//   overhang at >= T*s not produced         cache [s, C_out] to the first s
//                                         output samples; emit THIS chunk's
//                                         spill [T*s .. (T+1)*s) as OUTPUT
//
// Graph layout mirrors codec.cpp: ne0 = time, ne1 = channels (B=1). The
// persistent graph is allocated once per chunk_frames via its own gallocr and
// recomputed per chunk; per-chunk data enters as ggml INPUTS (latent, conv
// caches, tconv tail caches) and leaves as OUTPUTS (audio, new caches).
#include "codec_stream.hpp"

#include "backend.hpp"
#include "codec.hpp"
#include "common.hpp"
#include "model_loader.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// FSQ dequantize (host side) - identical to codec_fsq_dequantize, but writing
// into a caller-owned latent buffer so the padded final chunk decodes zeros.
// ---------------------------------------------------------------------------

namespace {

void fsq_into(const magpie_model& model, const int32_t* codes, int32_t n_frames,
              int32_t latent_frames, std::vector<float>& latent) {
    const magpie_codec_hparams& hp = model.hparams.codec;
    if (latent_frames < n_frames) {
        throw std::runtime_error("codec_stream: latent buffer smaller than frame count");
    }
    latent.assign((size_t)hp.latent_dim * latent_frames, 0.0f);
    if (n_frames <= 0) return;

    const int32_t n_groups = (int32_t)hp.fsq_num_groups;
    const int32_t dpg      = (int32_t)hp.fsq_num_levels.size();
    // codebook size = product of levels (matches codec.cpp fsq_codebook_size)
    int32_t cb = 1;
    for (int32_t l : hp.fsq_num_levels) cb *= l;

    for (int32_t g = 0; g < n_groups; ++g) {
        for (int32_t t = 0; t < n_frames; ++t) {
            const int32_t code = codes[(size_t)g * n_frames + t];
            if (code < 0 || code >= cb) {
                throw std::runtime_error("codec_stream: code id " +
                    std::to_string(code) + " out of range [0, " +
                    std::to_string(cb) + ") at group " + std::to_string(g) +
                    " frame " + std::to_string(t));
            }
            for (int32_t d = 0; d < dpg; ++d) {
                const int32_t L    = hp.fsq_num_levels[d];
                const int32_t base = hp.fsq_dim_base_index[d];
                const int32_t kq   = (code / base) % L;
                const int32_t half = L / 2;
                latent[((size_t)g * dpg + d) * latent_frames + t] =
                    (float)(kq - half) / (float)half;
            }
        }
    }
}

// One streaming cache: a small [len, channels] F32 buffer read as a graph
// INPUT and rewritten from a graph OUTPUT each chunk.
struct stream_cache {
    int64_t len      = 0;
    int64_t channels = 0;
    std::vector<float> data;

    void ensure(int64_t l, int64_t c) {
        if (len != l || channels != c || data.size() != (size_t)(l * c)) {
            len      = l;
            channels = c;
            data.assign((size_t)(l * c), 0.0f);
        }
    }
};

// Site tables. Order is fixed by graph-build order; both the input
// registration and the output readback use the same index so inputs and
// outputs stay paired across rebuilds.
enum cache_kind : int { CACHE_CONV = 0, CACHE_TCONV = 1 };

} // namespace

struct magpie_codec_stream_state::impl {
    std::vector<stream_cache> conv_caches;   // per causal-conv site
    std::vector<stream_cache> tconv_tails;   // per transposed-conv site
    size_t conv_pos  = 0;
    size_t tconv_pos = 0;

    void begin_graph() { conv_pos = 0; tconv_pos = 0; }
    void clear() {
        conv_caches.clear();
        tconv_tails.clear();
        begin_graph();
    }
};

magpie_codec_stream_state::magpie_codec_stream_state() : p(new impl()) {}
magpie_codec_stream_state::~magpie_codec_stream_state() = default;

std::vector<std::vector<float>> magpie_codec_stream_state::debug_caches() const {
    std::vector<std::vector<float>> out;
    for (const auto& c : p->conv_caches) out.push_back(c.data);
    for (const auto& c : p->tconv_tails) out.push_back(c.data);
    return out;
}

void magpie_codec_stream_state::clear() { p->clear(); }

struct magpie_codec_stream_graph::impl {
    ggml_context*  ctx    = nullptr;   // owns graph + tensor metadata
    ggml_cgraph*   gf     = nullptr;
    ggml_gallocr_t allocr = nullptr;
    int32_t        chunk_frames = 0;

    ggml_tensor* latent = nullptr;     // input  [chunk_frames, latent_dim]
    ggml_tensor* audio  = nullptr;     // output [chunk_frames * spf, 1]

    struct pending {
        ggml_tensor* t   = nullptr;
        int         kind = CACHE_CONV;
        size_t      idx  = 0;
    };
    struct host_input {
        ggml_tensor* t    = nullptr;
        const float* data = nullptr;
        host_input(ggml_tensor* t_, const float* d_) : t(t_), data(d_) {}
    };
    std::vector<pending> inputs;
    std::vector<pending> outputs;
    std::vector<host_input> host_inputs;         // alpha reciprocals etc.
    std::vector<std::unique_ptr<std::vector<float>>> staging;  // owns host data

    size_t output_samples   = 0;       // chunk_frames * samples_per_frame
    int64_t samples_per_frame = 0;

    std::vector<float> latent_data;    // staging (outlives compute)
    std::vector<float> audio_data;

    void free() {
        if (allocr) { ggml_gallocr_free(allocr); allocr = nullptr; }
        if (ctx)    { ggml_free(ctx);            ctx     = nullptr; }
        gf = nullptr;
        latent = nullptr;
        audio  = nullptr;
        inputs.clear();
        outputs.clear();
        host_inputs.clear();
        staging.clear();
        chunk_frames = 0;
        output_samples = 0;
        samples_per_frame = 0;
        latent_data.clear();
        audio_data.clear();
    }
};

magpie_codec_stream_graph::magpie_codec_stream_graph() : p(new impl()) {}
magpie_codec_stream_graph::~magpie_codec_stream_graph() { p->free(); }
void magpie_codec_stream_graph::reset() { p->free(); }
bool magpie_codec_stream_graph::initialized() const {
    return p->ctx && p->gf && p->allocr && p->latent && p->audio &&
           p->chunk_frames > 0;
}
int32_t magpie_codec_stream_graph::chunk_frames() const { return p->chunk_frames; }

// ---------------------------------------------------------------------------
// Streaming graph builders (mirrors of codec.cpp primitives with state I/O)
// ---------------------------------------------------------------------------

namespace {

// stream_cache accessor on the state, with shape (re)allocation.
stream_cache& cache_at(std::vector<stream_cache>& caches, size_t idx,
                       int64_t len, int64_t channels) {
    if (idx >= caches.size()) caches.resize(idx + 1);
    stream_cache& c = caches[idx];
    c.ensure(len, channels);
    return c;
}

ggml_tensor* cache_input(ggml_context* ctx,
                         magpie_codec_stream_state::impl& st,
                         std::vector<magpie_codec_stream_graph::impl::pending>& inputs,
                         int kind, size_t idx, int64_t len, int64_t channels) {
    std::vector<stream_cache>& caches = (kind == CACHE_CONV) ? st.conv_caches
                                                             : st.tconv_tails;
    cache_at(caches, idx, len, channels);
    ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, len, channels, 1);
    ggml_set_input(t);
    MG_LOG("codec_stream: cache IN  site=%zu kind=%s len=%d ch=%d", idx,
           kind == CACHE_CONV ? "conv" : "tconv", (int)len, (int)channels);
    inputs.push_back({t, kind, idx});
    return t;
}

void cache_output(ggml_tensor* t,
                  std::vector<magpie_codec_stream_graph::impl::pending>& outputs,
                  int kind, size_t idx) {
    ggml_set_output(t);
    MG_LOG("codec_stream: cache OUT site=%zu kind=%s", idx,
           kind == CACHE_CONV ? "conv" : "tconv");
    outputs.push_back({t, kind, idx});
}

// Causal conv1d with left-context cache. Mirrors codec.cpp causal_conv1d
// (f32 im2col + mul_mat) but pads with the cache instead of zeros and emits
// the new tail as an output.
ggml_tensor* stream_causal_conv1d(
        ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
        int dilation, magpie_codec_stream_state::impl& st,
        std::vector<magpie_codec_stream_graph::impl::pending>& inputs,
        std::vector<magpie_codec_stream_graph::impl::pending>& outputs) {
    const int k    = (int)w->ne[0];
    const int left = (k - 1) * dilation;

    ggml_tensor* conv_in = x;
    ggml_tensor* tail    = nullptr;
    if (left > 0) {
        const size_t idx = st.conv_pos++;
        ggml_tensor* cache = cache_input(ctx, st, inputs, CACHE_CONV, idx,
                                         left, x->ne[1]);
        conv_in = ggml_concat(ctx, cache, x, /*dim*/0);

        // new tail = last `left` frames of the conv INPUT. Build it from the
        // ORIGINAL tensors (cache and x), never from a view of the concat:
        // the tail is x's last min(left, T_x) frames, preceded (when
        // left > T_x) by the cache's last left-T_x frames.
        const int64_t T_x = x->ne[0];
        if (left <= T_x) {
            ggml_tensor* tv = ggml_view_3d(ctx, x, left, x->ne[1], 1,
                                           x->nb[1], x->nb[2],
                                           (size_t)(T_x - left) * x->nb[0]);
            tail = ggml_cont_3d(ctx, tv, left, x->ne[1], 1);
        } else {
            // left > T_x: the tail spans the cache's rows [T_x, left) (the
            // frames this chunk's x does not reach back over) plus ALL of
            // this chunk's x.
            ggml_tensor* t1 = ggml_view_3d(ctx, cache, left - T_x, x->ne[1], 1,
                                           cache->nb[1], cache->nb[2],
                                           (size_t)T_x * cache->nb[0]);
            tail = ggml_cont_3d(ctx, ggml_concat(ctx, t1, x, /*dim*/0),
                                left, x->ne[1], 1);
        }
        cache_output(tail, outputs, CACHE_CONV, idx);
    }

    // identical math to codec.cpp::causal_conv1d
    ggml_tensor* im = ggml_im2col(ctx, w, conv_in, /*s0*/1, /*s1*/0,
                                  /*p0*/0, /*p1*/0, /*d0*/dilation, /*d1*/0,
                                  /*is_2D*/false, GGML_TYPE_F32);
    ggml_tensor* im2 = ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2]);
    ggml_tensor* w2  = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor* y   = ggml_mul_mat(ctx, im2, w2);              // [T, C_out]
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
}

// Grouped causal ConvTranspose1d (groups == C_out, C_in == 2*C_out, k == 2*s)
// with spill caching. Offline (codec.cpp) the y[t-1] kernel half shifts right
// by s and the overhang beyond T*s is never produced. Here the first s output
// samples additionally receive the PREVIOUS chunk's overhang (cache INPUT,
// pre-added), and the overhang of THIS chunk is emitted as a cache OUTPUT.
ggml_tensor* stream_grouped_causal_tconv1d(
        ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
        int stride, magpie_codec_stream_state::impl& st,
        std::vector<magpie_codec_stream_graph::impl::pending>& inputs,
        std::vector<magpie_codec_stream_graph::impl::pending>& outputs) {
    const int64_t T     = x->ne[0];
    const int64_t c_in  = x->ne[1];
    const int64_t k     = w->ne[0];
    const int64_t c_out = b->ne[0];
    const int64_t s     = stride;
    if (w->ne[1] != 1 || w->ne[2] != c_in || c_in != 2 * c_out || k != 2 * s) {
        throw std::runtime_error("codec_stream: unsupported upsample conv geometry");
    }

    ggml_tensor* xr = ggml_reshape_3d(ctx, x, 1, T, c_in);      // rows of length 1
    ggml_tensor* half[2];                                       // [s*T, C_out] each
    for (int h = 0; h < 2; ++h) {
        ggml_tensor* wh = ggml_view_3d(ctx, w, 1, s, c_in,
                                       /*nb1*/ sizeof(float),
                                       /*nb2*/ (size_t)k * sizeof(float),
                                       /*offset*/ (size_t)h * s * sizeof(float));
        wh = ggml_cont(ctx, wh);
        ggml_tensor* pr = ggml_mul_mat(ctx, wh, xr);            // [s, T, C_in]
        ggml_tensor* v0 = ggml_view_3d(ctx, pr, s, T, c_out, pr->nb[1], 2 * pr->nb[2], 0);
        ggml_tensor* v1 = ggml_view_3d(ctx, pr, s, T, c_out, pr->nb[1], 2 * pr->nb[2], pr->nb[2]);
        half[h] = ggml_reshape_2d(ctx, ggml_add(ctx, v0, v1), s * T, c_out);
    }
    // second kernel half contributes to the NEXT frame: shift right by s.
    // Streamed variant: instead of padding `half[1]` to T*s + s and dropping
    // the overhang (offline), keep the FIRST s samples of `y` as-is and let
    // the tconv cache deliver the previous chunk's overhang; emit the last s
    // samples of the padded result as this chunk's overhang.
    ggml_tensor* bp = ggml_pad_ext(ctx, half[1], (int)s, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor* bs = ggml_view_2d(ctx, bp, s * T, c_out, bp->nb[1], 0);
    ggml_tensor* y  = ggml_add(ctx, half[0], bs);               // [s*T + s, C_out]

    // y has T*s samples (the add's view length); bp has T*s + s rows whose last
    // s rows = half[1]'s last s samples = THIS chunk's spill for the next chunk.
    const size_t idx = st.tconv_pos++;
    ggml_tensor* spill = cache_input(ctx, st, inputs, CACHE_TCONV, idx,
                                     s, c_out);

    // THIS chunk's spill = rows [s*T, s*T+s) of bp (= half1's last s samples).
    ggml_tensor* sv = ggml_view_3d(ctx, bp, s, c_out, 1, bp->nb[1], bp->nb[2],
                                   (size_t)(s * T) * bp->nb[0]);
    ggml_tensor* next_spill = ggml_cont_3d(ctx, sv, s, c_out, 1);

    // output[0..s) += previous spill (computed inside the graph)
    ggml_tensor* cur_view = ggml_view_3d(ctx, y, s, c_out, 1, y->nb[1], y->nb[2], 0);
    ggml_tensor* summed   = ggml_add(ctx, cur_view, spill);

    // full current-chunk output: [summed ; y[s .. s*T)]
    ggml_tensor* rest = nullptr;
    if (s * T > s) {
        rest = ggml_view_3d(ctx, y, s * (T - 1), c_out, 1, y->nb[1], y->nb[2],
                            (size_t)s * y->nb[0]);
        y = ggml_concat(ctx, summed, rest, /*dim*/0);
    } else {
        y = summed;   // T == 1: whole output is the summed prefix
    }
    cache_output(next_spill, outputs, CACHE_TCONV, idx);

    ggml_tensor* out = ggml_view_2d(ctx, y, s * T, c_out, y->nb[1], 0);
    return ggml_add(ctx, out, ggml_reshape_2d(ctx, b, 1, c_out));
}

// half_snake: identical to codec.cpp (kept local to avoid touching the
// offline file; same ops, same order, same eps handling via inv_alpha input).
ggml_tensor* stream_half_snake(ggml_context* ctx, ggml_tensor* x,
                               ggml_tensor* alpha, ggml_tensor* inv_alpha,
                               float slope) {
    const int64_t T   = x->ne[0];
    const int64_t C   = x->ne[1];
    const int64_t c_s = alpha->ne[1];                           // C // 2
    ggml_tensor* xs = ggml_view_2d(ctx, x, T, c_s, x->nb[1], 0);
    ggml_tensor* xl = ggml_view_2d(ctx, x, T, C - c_s, x->nb[1],
                                   (size_t)c_s * x->nb[1]);
    ggml_tensor* t = ggml_sin(ctx, ggml_mul(ctx, xs, alpha));
    t = ggml_sqr(ctx, t);
    t = ggml_mul(ctx, t, inv_alpha);
    ggml_tensor* snake_out = ggml_add(ctx, t, xs);
    ggml_tensor* lrelu_out = ggml_leaky_relu(ctx, xl, slope, /*inplace*/false);
    return ggml_concat(ctx, snake_out, lrelu_out, /*dim*/1);
}

} // namespace

// ---------------------------------------------------------------------------
// Left context
// ---------------------------------------------------------------------------

int32_t codec_stream_left_context_frames(const magpie_model& model) {
    const magpie_codec_hparams& hp = model.hparams.codec;
    // pre_conv (dilation 1) sees the raw frame domain; every deeper conv's
    // receptive field in FRAME units shrinks by the cumulative upsampling.
    // Total left context in samples over-approximates by the deepest domain;
    // convert back conservatively (ceil) so the cache always covers it.
    const int n_stages = (int)hp.up_sample_rates.size();
    // cumulative product of up-sample rates up to stage i, in frame domain
    int64_t cum = 1;
    int64_t ctx_samples = 0;
    // pre_conv: (k-1) * 1 frame-domain sample
    ctx_samples += (hp.in_kernel_size - 1);          // 6 frame-domain samples
    auto conv_frames = [&](int64_t k, int64_t dil, int64_t cum_rate) {
        return ((k - 1) * dil + cum_rate - 1) / cum_rate;   // ceil to frames
    };
    for (int i = 0; i < n_stages; ++i) {
        const int64_t rate = hp.up_sample_rates[i];
        cum *= rate;
        // upsample conv: kernel k == 2*rate, stride rate -> left ctx in its
        // OUTPUT domain is (k - rate) samples
        ctx_samples += std::max<int64_t>(0, hp.up_kernel_sizes[i] - rate);
        // res blocks (k in {3,7,11}, dil in {1,3,5}) at this stage
        int64_t worst = 0;
        for (int32_t kd : hp.resblock_kernel_sizes)
            for (int32_t dd : hp.resblock_dilation_sizes)
                worst = std::max<int64_t>(worst, (int64_t)(kd - 1) * dd);
        ctx_samples += worst;
    }
    // post_conv (k=3, dilation 1)
    ctx_samples += (hp.out_kernel_size - 1);
    const int64_t frames = (ctx_samples + cum - 1) / cum + 1; // ceil + 1 guard
    return (int32_t)frames;
}

// ---------------------------------------------------------------------------
// Init / decode
// ---------------------------------------------------------------------------

void codec_stream_init(const magpie_model& model, mg::backend& be,
                       int32_t chunk_frames,
                       magpie_codec_stream_state& state,
                       magpie_codec_stream_graph& graph) {
    const magpie_codec_hparams& hp = model.hparams.codec;
    if (chunk_frames <= 0) {
        throw std::runtime_error("codec_stream: chunk_frames must be positive");
    }
    graph.p->free();

    const int n_stages = (int)hp.up_sample_rates.size();
    for (int i = 0; i < n_stages; ++i) {
        if (hp.up_kernel_sizes[i] != 2 * hp.up_sample_rates[i]) {
            throw std::runtime_error("codec_stream: only k == 2*rate upsamplers supported");
        }
    }

    magpie_codec_stream_graph::impl& g = *graph.p;
    g.chunk_frames = chunk_frames;

    const size_t graph_nodes = 16384;
    const size_t buf_size = ggml_tensor_overhead() * graph_nodes +
                            ggml_graph_overhead_custom(graph_nodes, false);
    ggml_init_params ip{
        /*mem_size  =*/ buf_size,
        /*mem_buffer=*/ nullptr,
        /*no_alloc  =*/ true,
    };
    g.ctx = ggml_init(ip);
    if (!g.ctx) throw std::runtime_error("codec_stream: ggml_init failed");

    state.p->begin_graph();
    g.inputs.clear();
    g.outputs.clear();

    ggml_context* ctx = g.ctx;
    const std::string P = "codec.audio_decoder.";
    const int64_t T = chunk_frames;

    ggml_tensor* x0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, hp.latent_dim);
    ggml_set_input(x0);
    g.latent = x0;

    auto inv_alpha = [&](const std::string& alpha_name) {
        ggml_tensor* a = model.require_host_tensor(alpha_name);
        const int64_t n = ggml_nelements(a);
        g.staging.emplace_back(new std::vector<float>((size_t)n));
        const float* src = (const float*)a->data;
        float*       dst = g.staging.back()->data();
        for (int64_t i = 0; i < n; ++i) dst[i] = 1.0f / (src[i] + hp.snake_eps);
        ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n);
        ggml_set_input(t);
        g.host_inputs.emplace_back(t, dst);
        return t;
    };
    auto snake = [&](ggml_tensor* x, const std::string& alpha_name) {
        ggml_tensor* a = model.require_tensor(alpha_name);
        ggml_tensor* r = stream_half_snake(ctx, x, a, inv_alpha(alpha_name), hp.lrelu_slope);
        return r;
    };
    auto conv = [&](ggml_tensor* x, const std::string& path, int dilation) {
        return stream_causal_conv1d(ctx, x,
            model.require_tensor(path + ".conv.weight"),
            model.require_tensor(path + ".conv.bias"), dilation,
            *state.p, g.inputs, g.outputs);
    };
    auto tconv = [&](ggml_tensor* x, const std::string& path, int stride) {
        ggml_tensor* r = stream_grouped_causal_tconv1d(ctx, x,
            model.require_tensor(path + ".weight"),
            model.require_tensor(path + ".bias"), stride,
            *state.p, g.inputs, g.outputs);
        return r;
    };

    // pre_conv: [T, 32] -> [T, 864]
    MG_LOG("codec_stream: building graph for chunk_frames=%d", (int)chunk_frames);
    ggml_tensor* x = conv(x0, P + "pre_conv", 1);

    for (int i = 0; i < n_stages; ++i) {
        x = snake(x, P + "activations." + std::to_string(i) +
                     ".activation.snake_act.alpha");
        x = tconv(x, P + "up_sample_conv_layers." + std::to_string(i) + ".conv",
                  hp.up_sample_rates[i]);

        // HiFiGANResLayer: mean of 3 sequential res-block chains (k = 3,7,11)
        ggml_tensor* acc = nullptr;
        for (size_t j = 0; j < hp.resblock_kernel_sizes.size(); ++j) {
            ggml_tensor* xb = x;
            for (size_t m = 0; m < hp.resblock_dilation_sizes.size(); ++m) {
                const std::string blk = P + "res_layers." + std::to_string(i) +
                    ".res_blocks." + std::to_string(j) +
                    ".res_blocks." + std::to_string(m);
                ggml_tensor* h = snake(xb, blk + ".input_activation.activation.snake_act.alpha");
                h = conv(h, blk + ".input_conv", hp.resblock_dilation_sizes[m]);
                h = snake(h, blk + ".skip_activation.activation.snake_act.alpha");
                h = conv(h, blk + ".skip_conv", 1);
                xb = ggml_add(ctx, xb, h);                  // residual
            }
            acc = acc ? ggml_add(ctx, acc, xb) : xb;
        }
        x = ggml_scale(ctx, acc, 1.0f / (float)hp.resblock_kernel_sizes.size());
    }

    x = snake(x, P + "post_activation.activation.snake_act.alpha");
    x = conv(x, P + "post_conv", 1);                        // [1024*T + s?, 1]
    x = ggml_clamp(ctx, x, -1.0f, 1.0f);

    // The post_conv output can be up to samples_per_frame*T + spill samples
    // long depending on graph shape; the audio tensor is the clamped node.
    ggml_set_output(x);
    g.audio = x;

    g.gf = ggml_new_graph_custom(ctx, graph_nodes, false);
    for (const auto& pd : g.outputs) ggml_build_forward_expand(g.gf, pd.t);
    ggml_build_forward_expand(g.gf, g.audio);

    g.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be.handle()));
    if (!g.allocr) {
        g.free();
        throw std::runtime_error("codec_stream: gallocr_new failed");
    }
    if (!ggml_gallocr_alloc_graph(g.allocr, g.gf)) {
        g.free();
        throw std::runtime_error("codec_stream: graph allocation failed");
    }

    g.output_samples = (size_t)ggml_nelements(g.audio);
    if (g.output_samples == 0 ||
        g.output_samples % (size_t)chunk_frames != 0) {
        const size_t spf = hp.samples_per_frame;
        g.free();
        throw std::runtime_error("codec_stream: unexpected output length " +
                                 std::to_string(g.output_samples) +
                                 " for chunk_frames " + std::to_string(chunk_frames) +
                                 " (samples_per_frame " + std::to_string(spf) + ")");
    }
    g.samples_per_frame = (int64_t)(g.output_samples / (size_t)chunk_frames);
    g.latent_data.assign((size_t)chunk_frames * hp.latent_dim, 0.0f);
    g.audio_data.resize(g.output_samples);
}

std::vector<float> codec_stream_decode(const magpie_model& model,
                                       const int32_t* codes, int32_t n_frames,
                                       magpie_codec_stream_state& state,
                                       magpie_codec_stream_graph& graph,
                                       mg::backend& be) {
    const magpie_codec_hparams& hp = model.hparams.codec;
    magpie_codec_stream_graph::impl& g = *graph.p;
    if (!codes || n_frames <= 0) {
        throw std::runtime_error("codec_stream: invalid codes/n_frames");
    }
    if (g.chunk_frames <= 0) {
        throw std::runtime_error("codec_stream: graph not initialized");
    }
    if (n_frames > g.chunk_frames) {
        throw std::runtime_error("codec_stream: chunk has " +
                                 std::to_string(n_frames) +
                                 " frames > chunk_frames " +
                                 std::to_string(g.chunk_frames));
    }

    // latent for the padded chunk (zero frames decode as 0 == offline pad)
    fsq_into(model, codes, n_frames, g.chunk_frames, g.latent_data);

    // upload latent + all cache inputs
    ggml_backend_tensor_set(g.latent, g.latent_data.data(), 0,
                            ggml_nbytes(g.latent));
    for (const auto& hi : g.host_inputs) {
        ggml_backend_tensor_set(hi.t, hi.data, 0, ggml_nbytes(hi.t));
    }
    state.p->begin_graph();  // reset positions to replay the site order
    for (const auto& pi : g.inputs) {
        std::vector<stream_cache>& caches =
            (pi.kind == CACHE_CONV) ? state.p->conv_caches : state.p->tconv_tails;
        if (pi.idx >= caches.size()) {
            throw std::runtime_error("codec_stream: cache state missing for input " +
                                     std::to_string(pi.idx));
        }
        stream_cache& c = caches[pi.idx];
        if (!c.data.empty()) {
            ggml_backend_tensor_set(pi.t, c.data.data(), 0, ggml_nbytes(pi.t));
        }
    }

    be.set_n_threads(std::thread::hardware_concurrency() > 0
                         ? (int)std::thread::hardware_concurrency() : 4);
    if (ggml_backend_graph_compute(be.handle(), g.gf) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec_stream: graph compute failed");
    }

    // read back cache outputs first (updates state), then audio
    for (const auto& pd : g.outputs) {
        std::vector<stream_cache>& caches =
            (pd.kind == CACHE_CONV) ? state.p->conv_caches : state.p->tconv_tails;
        if (pd.idx >= caches.size()) {
            throw std::runtime_error("codec_stream: cache state missing for output " +
                                     std::to_string(pd.idx));
        }
        stream_cache& c = caches[pd.idx];
        if (!c.data.empty()) {
            ggml_backend_tensor_get(pd.t, c.data.data(), 0, ggml_nbytes(pd.t));
        }
    }
    ggml_backend_tensor_get(g.audio, g.audio_data.data(), 0,
                            ggml_nbytes(g.audio));

    const size_t keep = std::min(g.audio_data.size(),
                                 (size_t)n_frames * (size_t)g.samples_per_frame);
    if (std::getenv("MAGPIE_STREAM_DEBUG")) {
        std::fprintf(stderr, "[sdbg] graph nodes=%d chunk=%d frames_in=%d\n",
                     ggml_graph_n_nodes(g.gf), (int)g.chunk_frames, (int)n_frames);
        // Dump every cache after this chunk (site-order, one line each).
        for (size_t i = 0; i < state.p->conv_caches.size(); ++i) {
            const stream_cache& c = state.p->conv_caches[i];
            size_t zeros = 0;
            for (float v : c.data) if (v == 0.0f) ++zeros;
            std::fprintf(stderr, "[sdbg] conv site %zu len=%d ch=%d first=%.6f last=%.6f zeros=%zu/%zu\n",
                         i, (int)c.len, (int)c.channels,
                         c.data.empty() ? 0.f : c.data.front(),
                         c.data.empty() ? 0.f : c.data.back(),
                         zeros, c.data.size());
        }
        for (size_t i = 0; i < state.p->tconv_tails.size(); ++i) {
            const stream_cache& c = state.p->tconv_tails[i];
            std::fprintf(stderr, "[sdbg] tconv site %zu len=%d ch=%d first=%.6f last=%.6f\n",
                         i, (int)c.len, (int)c.channels,
                         c.data.empty() ? 0.f : c.data.front(),
                         c.data.empty() ? 0.f : c.data.back());
        }
    }
    return std::vector<float>(g.audio_data.begin(),
                              g.audio_data.begin() + (ptrdiff_t)keep);
}

std::vector<float> codec_stream_decode_all(const magpie_model& model,
                                           const int32_t* codes, int32_t n_frames,
                                           int32_t chunk_frames, mg::backend& be) {
    if (n_frames <= 0) return {};
    magpie_codec_stream_state state;
    magpie_codec_stream_graph graph;
    std::vector<float> audio;
    audio.reserve((size_t)n_frames * model.hparams.codec.samples_per_frame);

    int32_t cf = std::min<int32_t>(chunk_frames > 0 ? chunk_frames : 4, n_frames);
    codec_stream_init(model, be, cf, state, graph);
    const int32_t C = (int32_t)model.hparams.codec.fsq_num_groups;  // codebooks
    std::vector<int32_t> chunk_codes((size_t)C * cf);
    for (int32_t start = 0; start < n_frames; start += cf) {
        const int32_t n = std::min(cf, n_frames - start);
        // gather the chunk's codes as a contiguous [C][n] block (the input is
        // codebook-major [C][n_frames], so per-codebook rows are not
        // contiguous across chunk boundaries)
        for (int32_t c = 0; c < C; ++c)
            for (int32_t j = 0; j < n; ++j)
                chunk_codes[(size_t)c * n + j] =
                    codes[(size_t)c * n_frames + start + j];
        std::vector<float> chunk = codec_stream_decode(
            model, chunk_codes.data(), n, state, graph, be);
        audio.insert(audio.end(), chunk.begin(), chunk.end());
    }
    return audio;
}

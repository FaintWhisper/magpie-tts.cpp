// Flat C ABI over the C++ API. No exception may cross this boundary.
#include "magpie_tts_capi.h"
#include "magpie_tts.h"
#include "common.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <new>
#include <string>

struct magpie_tts_ctx {
    magpie_tts_context* inner = nullptr;
    std::string last_error;

    ~magpie_tts_ctx() { magpie_tts_free(inner); }
};

static float* synthesize_with_options(magpie_tts_ctx* ctx, const char* text,
                                      const magpie_tts_options& opts,
                                      int* out_n_samples) {
    if (out_n_samples) *out_n_samples = 0;
    if (!ctx) return nullptr;
    if (!ctx->inner) { ctx->last_error = "context has no model"; return nullptr; }
    if (!text)       { ctx->last_error = "text is NULL"; return nullptr; }
    try {
        std::vector<float> pcm = magpie_tts_synthesize(*ctx->inner, text, opts);
        float* out = (float*)std::malloc(pcm.size() * sizeof(float));
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        std::copy(pcm.begin(), pcm.end(), out);
        if (out_n_samples) *out_n_samples = (int)pcm.size();
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    } catch (...) {
        ctx->last_error = "unknown exception";
    }
    return nullptr;
}

extern "C" {

int magpie_tts_capi_abi_version(void) {
    return 3;
}

magpie_tts_ctx* magpie_tts_capi_load(const char* gguf_path) {
    if (!gguf_path || !*gguf_path) {
        MG_LOG("capi_load: NULL/empty model path");
        return nullptr;
    }
    magpie_tts_ctx* ctx = new (std::nothrow) magpie_tts_ctx();
    if (!ctx) return nullptr;
    try {
        ctx->inner = magpie_tts_load(gguf_path);
        return ctx;
    } catch (const std::exception& e) {
        MG_LOG("capi_load failed: %s", e.what());
    } catch (...) {
        MG_LOG("capi_load failed: unknown exception");
    }
    delete ctx;
    return nullptr;
}

void magpie_tts_capi_free(magpie_tts_ctx* ctx) {
    delete ctx;
}

float* magpie_tts_capi_synthesize(magpie_tts_ctx* ctx, const char* text,
                                  const char* language, const char* speaker,
                                  int* out_n_samples) {
    magpie_tts_options opts;
    if (language && *language) opts.language = language;
    if (speaker  && *speaker)  opts.speaker  = speaker;
    return synthesize_with_options(ctx, text, opts, out_n_samples);
}

float* magpie_tts_capi_synthesize_ex(
    magpie_tts_ctx* ctx, const char* text, const char* language,
    const char* speaker, uint64_t seed, float temperature, int topk,
    float cfg_scale, int n_threads, int max_frames, int* out_n_samples) {
    magpie_tts_options opts;
    if (language && *language) opts.language = language;
    if (speaker  && *speaker)  opts.speaker  = speaker;
    opts.seed = seed;
    opts.temperature = temperature;
    opts.topk = topk;
    opts.cfg_scale = cfg_scale;
    opts.n_threads = n_threads;
    opts.max_frames = max_frames;
    return synthesize_with_options(ctx, text, opts, out_n_samples);
}

// Bridge state for the C->C++ callback adapter. Lives on the caller's stack
// frame, which stays blocked for the whole streaming call while the codec
// worker invokes the callback.
struct stream_cb_bridge {
    magpie_tts_stream_pcm_cb cb;
    void* user;
    bool cancelled;
};

int magpie_tts_capi_synthesize_stream(
    magpie_tts_ctx* ctx, const char* text, const char* language,
    const char* speaker, uint64_t seed, float temperature, int topk,
    float cfg_scale, int n_threads, int max_frames, int chunk_frames,
    int codec_queue_depth, int n_threads_codec,
    magpie_tts_stream_pcm_cb pcm_cb, void* user,
    int* out_cancelled, double* out_ttfa_ms, int* out_chunk_frames,
    int* out_chunks, int* out_n_frames, long long* out_samples) {
    if (out_cancelled)   *out_cancelled   = 0;
    if (out_ttfa_ms)     *out_ttfa_ms     = -1.0;
    if (out_chunk_frames)*out_chunk_frames= 0;
    if (out_chunks)      *out_chunks      = 0;
    if (out_n_frames)    *out_n_frames    = 0;
    if (out_samples)     *out_samples     = 0;
    if (!ctx || !pcm_cb) return -1;
    if (!ctx->inner) { ctx->last_error = "context has no model"; return -1; }
    if (!text)       { ctx->last_error = "text is NULL";         return -1; }

    magpie_tts_options opts;
    if (language && *language) opts.language = language;
    if (speaker  && *speaker)  opts.speaker  = speaker;
    opts.seed = seed;
    opts.temperature = temperature;
    opts.topk = topk;
    opts.cfg_scale = cfg_scale;
    opts.n_threads = n_threads;
    opts.max_frames = max_frames;

    stream_cb_bridge bridge{pcm_cb, user, false};
    try {
        magpie_tts_stream_result res = magpie_tts_synthesize_stream(
            *ctx->inner, text ? text : "", opts, chunk_frames,
            codec_queue_depth, n_threads_codec,
            [&](const std::vector<uint8_t>& pcm) -> bool {
                // NOTE: `bridge` is captured by reference and outlives the
                // callback invocations (the streaming call is synchronous),
                // but the callback itself runs on the WORKER thread while
                // `bridge` lives on the CALLER's stack frame - safe because
                // that frame is blocked for the whole call.
                const bool keep = pcm_cb(pcm.data(), (int)pcm.size(), user) != 0;
                if (!keep) bridge.cancelled = true;
                return keep;
            });
        if (out_cancelled)    *out_cancelled    = res.cancelled ? 1 : 0;
        if (out_ttfa_ms)      *out_ttfa_ms      = res.stats.ttfa_ms;
        if (out_chunk_frames) *out_chunk_frames = res.stats.chunk_frames;
        if (out_chunks)       *out_chunks       = res.stats.chunks;
        if (out_n_frames)     *out_n_frames     = res.stats.n_frames;
        if (out_samples)      *out_samples      = (long long)res.stats.samples;
        if (bridge.cancelled || res.cancelled) return 1;
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    } catch (...) {
        ctx->last_error = "unknown exception";
    }
    return -1;
}

void magpie_tts_capi_free_string(char* s) {
    std::free(s);
}

void magpie_tts_capi_free_audio(float* samples) {
    std::free(samples);
}

const char* magpie_tts_capi_last_error(magpie_tts_ctx* ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

} // extern "C"

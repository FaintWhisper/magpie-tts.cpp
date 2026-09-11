#pragma once
// Streaming NanoCodec decode: decodes codebook frames incrementally in chunks
// while producing sample-exact output identical to the offline codec_decode()
// (see tests/test_codec_stream.cpp for the gate).
//
// Port of the state-carrying approach from NVIDIA's NeMo-Speech.cpp
// (src/tts/nanocodec/model.cpp, Apache-2.0), re-expressed over THIS repo's
// existing graph primitives so the numerical ops stay identical to the
// offline path:
//   * causal Conv1d: same f32 im2col + mul_mat pipeline as codec.cpp. The
//     offline left-pad (k-1)*dilation zero columns are replaced by a cache
//     INPUT holding the previous chunk's last (k-1)*dilation frames; the
//     new conv tail becomes a cache OUTPUT. Zero caches == zero pad, so the
//     first chunk is bit-identical to offline.
//   * causal grouped ConvTranspose1d (groups == C_out, C_in == 2*C_out,
//     k == 2*s): offline, the "x[t-1]" half is shifted right by s and the
//     contribution of x[T-1] to samples >= T*s is never materialized. In
//     streaming that "never materialized" overhang is exactly the state a
//     chunk owes the next one: the s-sample spill is emitted as a cache
//     OUTPUT and pre-added to the next chunk's transposed-conv prefix
//     (NVIDIA's nc_stream_causal_conv_transpose1d tail cache).
//
// State ownership: magpie_codec_stream_state holds only small host-side
// cache vectors; the graph shape is fixed by chunk_frames at init and
// reused (ggml graph rebuilt only when chunk_frames changes). The final
// partial chunk is zero-padded to chunk_frames; only the leading
// n_frames * samples_per_frame samples are returned.
//
// All frame counts here are CODEC frames (1 frame == 1024 samples @ 22050 Hz
// for this checkpoint; samples_per_frame comes from the GGUF hparams).
#include <cstdint>
#include <memory>
#include <vector>

struct magpie_model;
namespace mg { struct backend; }

// Host-side caches between chunks (conv left-context caches + transposed-conv
// tails). clear() returns the decoder to the pristine (zero-pad) state.
struct magpie_codec_stream_state {
    magpie_codec_stream_state();
    ~magpie_codec_stream_state();
    magpie_codec_stream_state(const magpie_codec_stream_state&) = delete;
    magpie_codec_stream_state& operator=(const magpie_codec_stream_state&) = delete;

    void clear();

    // Debug/testing access to the raw cache buffers (site order): conv caches
    // first, then transposed-conv tails. Test-only; not stable API.
    std::vector<std::vector<float>> debug_caches() const;

    struct impl;
    std::unique_ptr<impl> p;
};

// Persistent decode graph for a fixed chunk_frames. reset() frees it; a
// decode call with a different chunk_frames rebuilds automatically.
struct magpie_codec_stream_graph {
    magpie_codec_stream_graph();
    ~magpie_codec_stream_graph();
    magpie_codec_stream_graph(const magpie_codec_stream_graph&) = delete;
    magpie_codec_stream_graph& operator=(const magpie_codec_stream_graph&) = delete;

    void reset();
    bool initialized() const;
    int32_t chunk_frames() const;

    struct impl;
    std::unique_ptr<impl> p;
};

// Number of codec frames of left context the full causal decoder needs
// ((k-1)*dilation per conv site, accumulated over the upsample stack).
int32_t codec_stream_left_context_frames(const magpie_model& model);

// Build (or rebuild) the persistent graph for `chunk_frames` codec frames.
// Throws std::runtime_error on failure. `be` is the compute backend the
// model weights were uploaded to (same rules as codec_decode).
void codec_stream_init(const magpie_model& model, mg::backend& be,
                       int32_t chunk_frames,
                       magpie_codec_stream_state& state,
                       magpie_codec_stream_graph& graph);

// Decode one chunk. `codes` is codebook-major [C][n_frames], the same layout
// as codec_decode; n_frames must be in [1, chunk_frames]. Returns exactly
// n_frames * codec.samples_per_frame samples. Throws std::runtime_error.
std::vector<float> codec_stream_decode(const magpie_model& model,
                                       const int32_t* codes, int32_t n_frames,
                                       magpie_codec_stream_state& state,
                                       magpie_codec_stream_graph& graph,
                                       mg::backend& be);

// Convenience: whole-sequence streaming decode with fresh state. Equivalent
// (sample-for-sample) to codec_decode for n_frames >= 4; also handles short
// inputs by clamping the chunk size.
std::vector<float> codec_stream_decode_all(const magpie_model& model,
                                           const int32_t* codes, int32_t n_frames,
                                           int32_t chunk_frames, mg::backend& be);

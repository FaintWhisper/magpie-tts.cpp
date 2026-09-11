#ifndef MAGPIE_TTS_H
#define MAGPIE_TTS_H
// Public C++ API of magpie-tts.cpp (ggml port of nvidia MagpieTTS multilingual
// 357m + NanoCodec). For the flat C ABI (dlopen/purego), see magpie_tts_capi.h.

#ifdef __cplusplus
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Returns a static version string. Never null.
const char* magpie_tts_version();

// Opaque synthesis context: the loaded GGUF model plus (later) the tokenizer,
// KV caches and backend state, reused across synthesize calls.
struct magpie_tts_context;

// Engine-internal model view (src/model_loader.hpp). Exposed by reference so
// tools shipping their own engine code (CLI, tests) can reach the hparams and
// tensors without them being part of the public surface.
struct magpie_model;

// Per-phase wall-clock timing of one synthesis call, filled when
// magpie_tts_options::stats is set. magpie_tts_synthesize_codes fills every
// field except codec_ms (0 there; total_ms then excludes the codec);
// magpie_tts_synthesize adds the NanoCodec phase and the realtime factor.
struct magpie_tts_stats {
    double  encode_ms       = 0.0;  // tokenize + text encoder (once per call)
    double  decode_ms       = 0.0;  // AR decoder loop, INCLUDING lt_ms
    int32_t decode_steps    = 0;    // decoder steps executed (incl. EOS step)
    double  ms_per_step     = 0.0;  // decode_ms / decode_steps
    double  lt_ms           = 0.0;  // local-transformer share of decode_ms
    double  codec_ms        = 0.0;  // NanoCodec decode (codes -> PCM)
    double  total_ms        = 0.0;  // encode_ms + decode_ms + codec_ms
    double  audio_seconds   = 0.0;  // kept frames * samples_per_frame / rate
    double  realtime_factor = 0.0;  // audio_seconds / (total_ms / 1000)
};

struct magpie_tts_options {
    std::string language = "en";    // language_map key: en, de, es, fr, zh, hi, ja, ...
    std::string speaker  = "";      // baked speaker name (Aria, Jason, John, Leo,
                                    // Sofia); "" uses speaker_index
    int32_t     speaker_index = 0;  // 0..4, used when speaker is ""
    float       temperature = -1.0f;// < 0: model default (0.6)
    int32_t     topk        = -1;   // < 0: model default (80)
    float       cfg_scale   = -1.0f;// < 0: model default (2.5)
    uint64_t    seed        = 0;    // sampling seed; 0 = nondeterministic
    int32_t     n_threads   = 0;    // 0 = hardware concurrency
    int32_t     max_frames  = -1;   // max codec frames; < 0: model default (500)
    bool        use_cfg     = true; // classifier-free guidance (2 decoder streams)
    magpie_tts_stats* stats = nullptr;  // optional out: per-phase timing
};

// Codes-level synthesis result (magpie_tts_synthesize_codes).
struct magpie_tts_codes {
    // Codebook-major frame codes [C=8][n_frames] (codes[c*n_frames + t]),
    // EOS frames excluded -- ready for codec_decode.
    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    int32_t n_steps  = 0;  // decoder steps executed (including the EOS step)
    // When requested (magpie_tts_replay::collect_logits): the RAW (pre-CFG)
    // parallel-head logits of every step, step-major
    // [n_steps][n_stream][final_proj_dim].
    std::vector<float> step_logits;
};

// Teacher-forced replay configuration (parity testing): instead of sampling
// via the local transformer, feed these reference codes back through the
// decode loop. The prior evolution is driven by the replayed attention. The
// loop runs n_frames/frame_stacking + 1 steps (the +1 is the EOS step, whose
// codes are not in `codes`); EOS bookkeeping uses the argmax stream only.
struct magpie_tts_replay {
    const int32_t* codes = nullptr;  // codebook-major [8][n_frames]
    int32_t n_frames = 0;
    bool collect_logits = false;
    // false: full-sequence recompute each step with an empty KV cache, exactly
    // like the NeMo reference (earlier rows see the LATEST prior) -- slow,
    // for strict parity isolation only.
    bool use_kv_cache = true;
};

// Run the tokenizer + encoder + AR decode loop and return the frame codes
// (no codec). `replay` (optional) switches to teacher-forced replay.
// Throws std::runtime_error on failure.
magpie_tts_codes magpie_tts_synthesize_codes(magpie_tts_context& ctx,
                                             const std::string& text,
                                             const magpie_tts_options& options = {},
                                             const magpie_tts_replay* replay = nullptr);

// Load a GGUF model (arch "magpie-tts"). Throws std::runtime_error with a
// detailed message on failure (unreadable file, missing KV keys / tensors).
magpie_tts_context* magpie_tts_load(const std::string& gguf_path);

// Free a context. Safe on nullptr.
void magpie_tts_free(magpie_tts_context* ctx);

// The loaded model (hparams + tensor map). Valid as long as `ctx` lives.
const magpie_model& magpie_tts_model(const magpie_tts_context& ctx);

// Output sample rate (22050 for this checkpoint).
int32_t magpie_tts_sample_rate(const magpie_tts_context& ctx);

// Synthesize `text` to mono PCM f32 at magpie_tts_sample_rate() (22050 Hz),
// samples in [-1, 1]. Throws std::runtime_error on failure.
std::vector<float> magpie_tts_synthesize(magpie_tts_context& ctx,
                                         const std::string& text,
                                         const magpie_tts_options& options = {});

// ---------------------------------------------------------------------------
// Optional streaming path (Q8/f32 alike; changes WHEN audio is produced, not
// its format). The AR generator runs on the calling thread and pushes codebook
// frames to a bounded queue; a codec worker thread decodes fixed-size chunks
// with carried state (no clicks at chunk boundaries) and delivers PCM through
// `callback` before sentence generation finishes. Offline
// magpie_tts_synthesize() is unchanged and stays the comparison/rollback path.
// ---------------------------------------------------------------------------

// Called from the codec worker with interleaved mono PCM16 LE bytes
// (2 bytes/sample, [-32767, 32767] from a [-1, 1] clamp). Returns false to
// cancel the synthesis (queued audio is discarded, generation stops).
using magpie_pcm_callback = std::function<bool(const std::vector<uint8_t>&)>;

struct magpie_tts_stream_stats {
    double  ttfa_ms        = 0.0;  // start -> first PCM callback
    int32_t chunk_frames   = 0;    // codec frames per chunk actually used
    int32_t chunks         = 0;    // codec chunks decoded
    int32_t n_frames       = 0;    // codec frames generated (kept, EOS trimmed)
    uint64_t samples       = 0;    // samples delivered to the callback
    double  total_ms       = 0.0;  // wall clock of the whole call
};

// Streaming synthesis of `text`. Same inputs as magpie_tts_synthesize plus:
//   chunk_frames        codec frames per chunk (default 4 ~= 186 ms @ 22050);
//                       clamped to [1, 32].
//   codec_queue_depth   max chunks buffered ahead of the codec worker
//                       (default 4). The producer blocks when full, bounding
//                       VRAM working set.
//   n_threads_codec     CPU threads for the codec worker when it runs on CPU
//                       (0 = hardware concurrency). The worker always uses
//                       its OWN backend (a second CUDA stream context when
//                       MAGPIE_DEVICE=cuda) so it can overlap the AR loop.
// `callback` is invoked from the worker thread. Throws std::runtime_error on
// failure; on callback-false cancellation it returns stats with
// cancelled=true and does NOT throw.
struct magpie_tts_stream_result {
    bool cancelled = false;
    magpie_tts_stream_stats stats;
};
magpie_tts_stream_result magpie_tts_synthesize_stream(
    magpie_tts_context& ctx, const std::string& text,
    const magpie_tts_options& options, int32_t chunk_frames,
    int32_t codec_queue_depth, int32_t n_threads_codec,
    const magpie_pcm_callback& callback);

// Streaming decode of ALREADY-GENERATED codes (validation path; no AR loop).
// Equivalent to codec_decode but chunked; see tests/test_codec_stream.cpp.
std::vector<float> magpie_tts_decode_codes_stream(
    magpie_tts_context& ctx, const int32_t* codes, int32_t n_frames,
    int32_t chunk_frames);
#endif // __cplusplus

#endif // MAGPIE_TTS_H

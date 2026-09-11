// Gate: streaming NanoCodec decode == offline codec_decode, sample-for-sample.
//
// Runs the same integer code tokens through both paths and compares the
// waveforms. Chunking must not change the audio: the stream state carries
// every conv left-context and transposed-conv spill between chunks, so the
// concatenated chunk outputs are bit-identical to the offline decode (same
// ops, same order; caches zero-start == offline zero pad).
//
// Run: MAGPIE_MODEL=<q8.gguf> ctest -R test_codec_stream --output-on-failure
// (skips with code 77 when MAGPIE_MODEL is unset).
#include "codec.hpp"
#include "codec_stream.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
// Crash tracer with a symbolized stack walk (no debugger available here).
static LONG WINAPI crash_tracer(EXCEPTION_POINTERS* ep) {
    HMODULE mod = nullptr;
    char name[MAX_PATH] = {0};
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)ep->ExceptionRecord->ExceptionAddress, &mod)) {
        GetModuleFileNameA(mod, name, MAX_PATH);
    }
    std::fprintf(stderr, "\n[crash] code=0x%08lX addr=%p module=%s base=%p\n",
                 (unsigned long)ep->ExceptionRecord->ExceptionCode,
                 ep->ExceptionRecord->ExceptionAddress, name, (void*)mod);
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    if (SymInitialize(GetCurrentProcess(), NULL, TRUE)) {
        void* stack[32];
        USHORT n = CaptureStackBackTrace(0, 32, stack, NULL);
        for (USHORT i = 0; i < n; ++i) {
            SYMBOL_INFO* sym = (SYMBOL_INFO*)calloc(1, sizeof(SYMBOL_INFO) + 128);
            sym->MaxNameLen = 127;
            sym->SizeOfStruct = sizeof(SYMBOL_INFO);
            DWORD64 off = 0;
            if (SymFromAddr(GetCurrentProcess(), (DWORD64)stack[i], &off, sym)) {
                std::fprintf(stderr, "  #%02u %s + 0x%llx\n", i, sym->Name,
                             (unsigned long long)off);
            } else {
                std::fprintf(stderr, "  #%02u %p\n", i, stack[i]);
            }
            free(sym);
        }
    }
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
struct crash_tracer_init {
    crash_tracer_init() { AddVectoredExceptionHandler(0, &crash_tracer); }
} g_crash_tracer_init;
#endif
#include <algorithm>

int main() {
    const char* path = std::getenv("MAGPIE_MODEL");
    if (!path || !*path) {
        std::printf("MAGPIE_MODEL not set: skipping (exit 77)\n");
        return 77;
    }

    mg::backend be;
    be.init();
    std::printf("backend: %s\n", be.device_name());

    magpie_model model;
    model.load(path);
    model.upload_weights(be.handle());

    const int32_t C     = (int32_t)model.hparams.audio.num_codebooks;      // 8
    const int32_t S     = (int32_t)model.hparams.codec.samples_per_frame;  // 1024
    const int32_t n_frames = 64;   // 6 full chunks of 11 + partial, plus tails
    const int32_t cb_size  = [&]{
        int32_t n = 1;
        for (int32_t l : model.hparams.codec.fsq_num_levels) n *= l;
        return n;
    }();

    // Deterministic pseudo-random codes in [0, cb_size)
    std::vector<int32_t> codes((size_t)C * n_frames);
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (auto& c : codes) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        c = (int32_t)(rng % (uint64_t)cb_size);
    }

    // ---- offline reference ----
    std::printf("[trace] offline decode n_frames=%d\n", n_frames); std::fflush(stdout);
    std::vector<float> ref = codec_decode(model, codes.data(), n_frames, &be);
    std::printf("[trace] offline done, %zu samples\n", ref.size()); std::fflush(stdout);
    assert(ref.size() == (size_t)n_frames * S);

    // ---- streaming: several chunk sizes, incl. ones not dividing n_frames ----
    const int32_t sizes[] = {64, 32, 16, 4, 7};
    double worst = 0.0;
    for (int32_t cf : sizes) {
        std::printf("[trace] stream chunk_frames=%d init\n", cf); std::fflush(stdout);
        magpie_codec_stream_state st;
        magpie_codec_stream_graph gr;
        codec_stream_init(model, be, cf, st, gr);
        std::printf("[trace] stream chunk_frames=%d init done\n", cf); std::fflush(stdout);

        std::vector<float> got;
        got.reserve(ref.size());
        for (int32_t start = 0; start < n_frames; start += cf) {
            const int32_t n = std::min(cf, n_frames - start);
            // gather the chunk's codes as a contiguous [C][n] block (the
            // full-sequence layout is [C][n_frames], so per-codebook rows are
            // NOT contiguous across the chunk boundary)
            std::vector<int32_t> chunk_codes((size_t)C * n);
            for (int32_t c = 0; c < C; ++c)
                for (int32_t j = 0; j < n; ++j)
                    chunk_codes[(size_t)c * n + j] =
                        codes[(size_t)c * n_frames + start + j];
            std::vector<float> chunk = codec_stream_decode(
                model, chunk_codes.data(), n, st, gr, be);
            assert(chunk.size() == (size_t)n * S);
            got.insert(got.end(), chunk.begin(), chunk.end());
        }
        assert(got.size() == ref.size());

        double maxd = 0.0;
        size_t first_bad = SIZE_MAX;
        for (size_t i = 0; i < ref.size(); ++i) {
            const double d = (double)std::fabs((double)ref[i] - (double)got[i]);
            if (d > maxd) { maxd = d; first_bad = i; }
        }
        if (maxd > 0.1) {
            // per-chunk max diff profile
            for (int32_t ck = 0; ck * cf < n_frames; ++ck) {
                double cd = 0.0; size_t cb = 0;
                const int32_t n = std::min(cf, n_frames - ck * cf);
                for (int32_t j = 0; j < n * S; ++j) {
                    const double d = (double)std::fabs(
                        (double)ref[(size_t)ck * cf * S + j] - (double)got[(size_t)ck * cf * S + j]);
                    if (d > cd) { cd = d; cb = j; }
                }
                std::printf("  [prof] chunk %d maxd=%.4g at in-chunk sample %zu\n", ck, cd, cb);
                if (ck == 1 && cd > 0.1) {
                    int shown = 0;
                    for (int32_t j = 0; j < n * S && shown < 5; ++j) {
                        const double d = (double)std::fabs(
                            (double)ref[(size_t)ck * cf * S + j] - (double)got[(size_t)ck * cf * S + j]);
                        if (d > 1e-4) {
                            std::printf("    first-diffs: sample %d d=%.4g\n", j, d);
                            ++shown;
                            j += 511;  // skip ahead to find the onset region
                        }
                    }
                }
            }
            std::fflush(stdout);
        }
        if (cf == 4 && maxd > 0.1) {
            // one-shot: decode frames 20..24 (chunk 5) fresh with chunk_frames=64
            magpie_codec_stream_state st2;
            magpie_codec_stream_graph gr2;
            codec_stream_init(model, be, 64, st2, gr2);
            std::vector<int32_t> c5_codes((size_t)C * 4);
            for (int32_t c = 0; c < C; ++c)
                for (int32_t j = 0; j < 4; ++j)
                    c5_codes[(size_t)c * 4 + j] = codes[(size_t)c * n_frames + 20 + j];
            std::vector<float> c5 = codec_stream_decode(
                model, c5_codes.data(), 4, st2, gr2, be);
            double d5 = 0.0;
            for (int i = 0; i < 4 * S; ++i)
                d5 = std::max(d5, (double)std::fabs((double)ref[(size_t)20 * S + i] - (double)c5[i]));
            std::printf("[diag] fresh cf=64 decode of chunk5 codes vs ref: %.3g\n", d5);
            std::fflush(stdout);
        }
        if (cf == 64 || cf == 32) {
            // probe: pre_conv cache must equal the latent of the last 6 frames.
            // Build a contiguous [C][6] block first (codec_fsq_dequantize takes
            // codebook-major codes with the block's own stride).
            const int32_t tail_start = n_frames - 6;
            std::vector<int32_t> tail_codes((size_t)C * 6);
            for (int32_t c = 0; c < C; ++c)
                for (int32_t j = 0; j < 6; ++j)
                    tail_codes[(size_t)c * 6 + j] =
                        codes[(size_t)c * n_frames + tail_start + j];
            std::vector<float> lat = codec_fsq_dequantize(model, tail_codes.data(), 6);
            const std::vector<float> c0 = st.debug_caches().at(0);
            double d = 0.0;
            for (int i = 0; i < 6 * 32; ++i)
                d = std::max(d, (double)std::fabs((double)lat[i] - (double)c0[i]));
            std::printf("[probe] pre_conv cache vs latent(58..63): %.3g (cache0=%.6f lat0=%.6f)\n", d,
                        c0.empty() ? 0.f : c0[0], lat.empty() ? 0.f : lat[0]);
            // find first mismatching element
            for (int i = 0; i < 6 * 32; ++i) {
                if ((double)std::fabs((double)lat[i] - (double)c0[i]) > 1e-6) {
                    std::printf("[probe] first mismatch at flat %d (frame %d, ch %d) lat=%.6f cache=%.6f\n",
                                i, i / 32, i % 32, lat[i], c0[i]);
                    break;
                }
            }
            // raw dump of first 12 elements of both
            std::printf("[probe] lat  first12:");
            for (int i = 0; i < 12; ++i) std::printf(" %.3f", lat[i]);
            std::printf("\n[probe] cache first12:");
            for (int i = 0; i < 12; ++i) std::printf(" %.3f", c0[i]);
            std::printf("\n");
            // dump 18..42 of both
            std::printf("[probe] lat  [18..42):");
            for (int i = 18; i < 42; ++i) std::printf(" %.3f", lat[i]);
            std::printf("\n[probe] cache[18..42):");
            for (int i = 18; i < 42; ++i) std::printf(" %.3f", c0[i]);
            std::printf("\n");
            std::fflush(stdout);
            // how much of the cache matches latent(58..63) per frame?
            for (int f = 0; f < 6; ++f) {
                double df = 0.0;
                for (int c2 = 0; c2 < 32; ++c2)
                    df = std::max(df, (double)std::fabs(
                        (double)lat[(size_t)c2 * 6 + f] - (double)c0[(size_t)f + c2 * 6]));
                std::printf("[probe]   tail frame %d maxdiff %.3g\n", f, df);
            }
            // site 1 cache (pre_conv output tail, left=2) vs offline dump rows 30..31
            std::printf("[probe] site1 env=%d\n", std::getenv("MAGPIE_CODEC_DUMP_FILE") ? 1 : 0);
            if (cf == 32 && std::getenv("MAGPIE_CODEC_DUMP_FILE")) {
                std::vector<float> c1 = st.debug_caches().at(1);
                FILE* f = std::fopen(std::getenv("MAGPIE_CODEC_DUMP_FILE"), "rb");
                if (f) {
                    std::vector<float> pc(64 * 864);
                    if (std::fread(pc.data(), sizeof(float), pc.size(), f) == pc.size()) {
                        double d1 = 0.0;
                        for (int c2 = 0; c2 < 864; ++c2)
                            for (int f2 = 0; f2 < 2; ++f2)
                                d1 = std::max(d1, (double)std::fabs(
                                    (double)pc[(size_t)(30 + f2) * 864 + c2] -
                                    (double)c1[(size_t)f2 + c2 * 2]));
                        std::printf("[probe] site1 cache vs offline pre_conv rows 30..31: %.3g\n", d1);
                        std::printf("[probe]   c1[0..6]:");
                        for (int i = 0; i < 6; ++i) std::printf(" %.4g", c1[i]);
                        std::printf("\n[probe]   pc[30*864..+6]:");
                        for (int i = 0; i < 6; ++i) std::printf(" %.4g", pc[(size_t)30 * 864 + i]);
                        std::printf("\n");
                    }
                    std::fclose(f);
                } else {
                    std::printf("[probe] site1: fopen failed\n");
                }
            }
            std::fflush(stdout);
        }
        worst = std::max(worst, maxd);
        const double tol = be.gpu() ? 5e-3 : 1e-4;   // CUDA kernel reduction-order noise
        std::printf("chunk_frames=%3d  max|diff| = %.3g at sample %zu (chunk %zu)  (%s)\n",
                    cf, maxd, first_bad,
                    first_bad / ((size_t)cf * S),
                    maxd == 0.0 ? "BIT-EXACT" :
                    maxd < 1e-5 ? "within f32 tolerance" :
                    maxd < tol ? "within GPU kernel tolerance" : "MISMATCH");
        if (maxd >= tol) {
            std::fprintf(stderr, "FAIL: streaming != offline at chunk_frames=%d\n", cf);
            return 1;
        }
    }

    // ---- decode_all convenience path must agree too ----
    {
        std::vector<float> got = codec_stream_decode_all(model, codes.data(),
                                                         n_frames, 9, be);
        double maxd = 0.0;
        for (size_t i = 0; i < ref.size() && i < got.size(); ++i)
            maxd = std::max(maxd, (double)std::fabs((double)ref[i] - (double)got[i]));
        std::printf("decode_all(9)   max|diff| = %.3g\n", maxd);
        const double tol9 = be.gpu() ? 5e-3 : 1e-4;
        if (got.size() != ref.size() || maxd >= tol9) {
            std::fprintf(stderr, "FAIL: decode_all mismatch\n");
            return 1;
        }
    }

    std::printf("PASS: streaming == offline (worst max|diff| = %.3g)\n", worst);
    return 0;
}

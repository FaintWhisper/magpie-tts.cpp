#include "magpie_tts_capi.h"
#include <cstdio>
#include <vector>

static int g_calls = 0;
static std::vector<unsigned char> g_all;

static int pcm_cb(const unsigned char* data, int n, void* user) {
    (void)user;
    ++g_calls;
    g_all.insert(g_all.end(), data, data + n);
    return 1;  // keep going
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: test_capi_stream <gguf>\n"); return 2; }
    printf("abi=%d\n", magpie_tts_capi_abi_version());
    magpie_tts_ctx* ctx = magpie_tts_capi_load(argv[1]);
    if (!ctx) { printf("load failed\n"); return 1; }
    int cancelled = 0, cf = 0, chunks = 0, frames = 0;
    double ttfa = -1; long long samples = 0;
    int rc = magpie_tts_capi_synthesize_stream(
        ctx, "Hola, esta es una prueba del flujo de streaming.", "es", "Aria",
        1234, 0.6f, 80, 2.5f, 8, 500, 4, 4, 0, pcm_cb, nullptr,
        &cancelled, &ttfa, &cf, &chunks, &frames, &samples);
    printf("rc=%d cancelled=%d ttfa=%.1fms cf=%d chunks=%d frames=%d samples=%lld calls=%d bytes=%zu\n",
           rc, cancelled, ttfa, cf, chunks, frames, samples, g_calls, g_all.size());
    magpie_tts_capi_free(ctx);
    return rc == 0 && g_calls > 0 && samples > 0 ? 0 : 1;
}

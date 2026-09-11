# Symbolize an RVA in a PE using dbghelp (single-file helper).
param([string]$Exe = "C:\Workspace\magpie-tts.cpp\build-cuda\tests\test_codec_stream.exe", [string]$RvaHex = "39FB")
$src = Join-Path $env:TEMP "sym.cpp"
$code = @'
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: sym <exe> <rva-hex>\n"); return 2; }
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    if (!SymInitialize(GetCurrentProcess(), NULL, FALSE)) { printf("init fail\n"); return 1; }
    DWORD64 base = SymLoadModuleEx(GetCurrentProcess(), NULL, argv[1], NULL, 0x180000000, 0, NULL, 0);
    if (!base) { printf("load fail %lu\n", GetLastError()); return 1; }
    DWORD64 rva = strtoull(argv[2], NULL, 16);
    char buf[sizeof(SYMBOL_INFO) + 256] = {0};
    SYMBOL_INFO* sym = (SYMBOL_INFO*)buf;
    sym->MaxNameLen = 255; sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    DWORD64 off = 0;
    if (SymFromAddr(GetCurrentProcess(), base + rva, &off, sym))
        printf("%s + 0x%llx\n", sym->Name, (unsigned long long)off);
    else
        printf("no symbol (err %lu)\n", GetLastError());
    return 0;
}
'@
Set-Content -Path $src -Value $code -Encoding ASCII
$cl = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"
$libs = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
& $cl /nologo /O2 $src /Fe:"$env:TEMP\sym.exe" /I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt" /I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared" /I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um" dbghelp.lib /link /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\ucrt\x64" /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um\x64" /LIBPATH:"$libs" 2>&1 | Select-Object -Last 3
& "$env:TEMP\sym.exe" $Exe $RvaHex

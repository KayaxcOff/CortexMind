//
// Created by muham on 12.04.2026.
//

#include "CortexMind/tools/is_avx2_available.hpp"
#include <intrin.h>

using namespace cortex;

bool cortex::is_avx2_available() noexcept {
    int info[4];
    __cpuid(info, 1);

    const bool avx = info[2] & (1 << 28);
    bool osx_save = info[2] & (1 << 27);

    if (!avx || !osx_save) {
        return false;
    }

    unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return false;
    }

    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
}

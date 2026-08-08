//
// Created by muham on 8.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/reduce.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Engine/AVX2/horizontal.hpp>

using namespace cortex::_fw::avx2;

void reduce::sum(const float *Xx, float *Xz, const std::size_t N) {
    std::size_t i = 0;
    vec8f acc = zerof();
    for (; i + 8 <= N; i += 8) {
        acc = add(acc, loadu(Xx + i));
    }
    float result = horizontal::sum(acc);
    for (; i < N; ++i) {
        result += Xx[i];
    }
    Xz[0] = result;
}

void reduce::sum(const float *Xx, float *Xz, const std::size_t outer_size, const std::size_t dim_size, const std::size_t inner_size) {
    if (inner_size == 1) {
        for (std::size_t i = 0; i < outer_size; ++i) {
            const float* src = Xx + (i * dim_size);
            sum(src, Xz, dim_size);
        }
        return;
    }

    for (std::size_t i = 0; i < outer_size; ++i) {
        float* dst = Xz + (i * inner_size);
        const float* src1 = Xx + (i * dim_size * inner_size);

        std::size_t j = 0;
        for (; j + 8 <= dim_size; j += 8) {
            storeu(dst + j, loadu(src1 + j));
        }
        for (; j < inner_size; ++j) {
            dst[j] = src1[j];
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const float* src2 = Xx + ((i * dim_size + d) * inner_size);
            j = 0;
            for (; j + 8 <= inner_size; j += 8) {
                const vec8f out_val = loadu(dst + j);
                const vec8f in_val  = loadu(src2 + j);
                storeu(dst + j, add(out_val, in_val));
            }
            for (; j < inner_size; ++j) {
                dst[j] += src2[j];
            }
        }
    }
}

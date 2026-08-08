//
// Created by muham on 8.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/reduce.hpp"
#include <CortexMind/framework/Engine/AVX2/fma.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Engine/AVX2/horizontal.hpp>
#include <cmath>
#include <vector>

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

void reduce::mean(const float *Xx, float *Xz, const std::size_t N) {
    sum(Xx, Xz, N);
    Xz[0] /= static_cast<float>(N);
}

void reduce::mean(const float *Xx, float *Xz, const std::size_t outer_size, const std::size_t dim_size, const std::size_t inner_size) {
    sum(Xx, Xz, outer_size, dim_size, inner_size);

    const std::size_t total_size = outer_size * inner_size;
    std::size_t i = 0;

    const vec8f v_scale = set1(1.0f / static_cast<float>(dim_size));
    for (; i < total_size; ++i) {
        storeu(Xz + i, mul(loadu(Xz + i), v_scale));
    }
    for (; i < total_size; ++i) {
        Xz[i] /= static_cast<float>(dim_size);
    }
}

void reduce::var(const float *Xx, float *Xz, const std::size_t N) {
    mean(Xx, Xz, N);

    const float mu = Xz[0];
    const vec8f vmu = set1(mu);

    std::size_t i = 0;
    vec8f acc = zerof();
    for (; i + 8 <= N; i += 8) {
        const vec8f diff = sub(loadu(Xx + i), vmu);
        acc = fma::add(diff, diff, acc);
    }
    float result = horizontal::mean(acc);
    for (; i < N; ++i) {
        const float diff = Xx[i] - mu;
        result += diff * diff;
    }
    Xz[0] = result / static_cast<float>(N);
}

void reduce::var(const float *Xx, float *Xz, const std::size_t outer_size, const std::size_t dim_size, const std::size_t inner_size) {
    if (inner_size == 1) {
        for (std::size_t o = 0; o < outer_size; ++o) {
            const float* src = Xx + (o * dim_size);

            std::size_t i = 0;
            vec8f acc_sum = zerof();
            for (; i + 8 <= dim_size; i += 8) {
                acc_sum = add(acc_sum, loadu(src + i));
            }
            float sum_val = horizontal::sum(acc_sum);
            for (; i < dim_size; ++i) {
                sum_val += src[i];
            }
            const float mu = sum_val / static_cast<float>(dim_size);
            const vec8f vmu = set1(mu);

            i = 0;
            vec8f acc_var = zerof();
            for (; i + 8 <= dim_size; i += 8) {
                const vec8f diff = sub(loadu(src + i), vmu);
                acc_var = fma::add(diff, diff, acc_var);
            }
            float var_val = horizontal::sum(acc_var);
            for (; i < dim_size; ++i) {
                const float diff = src[i] - mu;
                var_val += diff * diff;
            }
            Xz[o] = var_val / static_cast<float>(dim_size);
        }
        return;
    }

    std::vector<float> temp_mean(outer_size * inner_size);
    mean(Xx, temp_mean.data(), outer_size, dim_size, inner_size);

    for (std::size_t o = 0; o < outer_size; ++o) {
        float* dst = Xz + (o * inner_size);
        const float* src_0 = Xx + (o * dim_size * inner_size);
        const float* mu_ptr = temp_mean.data() + (o * inner_size);

        std::size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            const vec8f vmu = loadu(mu_ptr + i);
            const vec8f diff = sub(loadu(src_0 + i), vmu);
            storeu(dst + i, mul(diff, diff));
        }
        for (; i < inner_size; ++i) {
            const float diff = src_0[i] - mu_ptr[i];
            dst[i] = diff * diff;
        }

        for (std::size_t d = 1; d < dim_size; ++d) {
            const float* src = Xx + ((o * dim_size + d) * inner_size);
            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f vmu = loadu(mu_ptr + i);
                const vec8f out_val = loadu(dst + i);
                const vec8f diff = sub(loadu(src + i), vmu);
                storeu(dst + i, fma::add(diff, diff, out_val));
            }
            for (; i < inner_size; ++i) {
                const float diff = src[i] - mu_ptr[i];
                dst[i] += diff * diff;
            }
        }
    }

    const size_t total_output_elements = outer_size * inner_size;
    size_t i = 0;
    const vec8f v_scale = set1(1.0f / static_cast<float>(dim_size));
    for (; i + 8 <= total_output_elements; i += 8) {
        storeu(Xz + i, mul(loadu(Xz + i), v_scale));
    }
    for (; i < total_output_elements; ++i) {
        Xz[i] /= static_cast<float>(dim_size);
    }
}

void reduce::stdv(const float *Xx, float *Xz, const std::size_t N) {
    var(Xx, Xz, N);
    Xz[0] = std::sqrt(Xx[0]);
}

void reduce::stdv(const float *Xx, float *Xz, const std::size_t outer_size, const std::size_t dim_size, const std::size_t inner_size) {
    var(Xx, Xz, outer_size, dim_size, inner_size);

    const size_t total_output_elements = outer_size * inner_size;
    std::size_t i = 0;

    for (; i + 8 <= total_output_elements; i += 8) {
        storeu(Xz + i, avx2::sqrt(loadu(Xz + i)));
    }
    for (; i < total_output_elements; ++i) {
        Xz[i] = std::sqrt(Xz[i]);
    }
}

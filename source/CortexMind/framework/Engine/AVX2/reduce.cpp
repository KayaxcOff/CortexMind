//
// Created by muham on 8.05.2026.
//

#include "CortexMind/framework/Engine/AVX2/reduce.hpp"
#include <CortexMind/framework/Engine/AVX2/cmp.hpp>
#include <CortexMind/framework/Engine/AVX2/fma.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Engine/AVX2/horizontal.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <utility>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

void reduce::sum(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = add(acc, loadu(Xx + i));
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += Xx[i];
    }
    Xz[0] = output;
}

void reduce::sum(const f32 *Xx, f32 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    if (inner_size == 1) {
        for (size_t o = 0; o < outer_size; ++o) {
            const f32* src = Xx + (o * dim_size);
            size_t i = 0;
            vec8f acc = zero();

            for (; i + 8 <= dim_size; i += 8) {
                acc = add(acc, loadu(src + i));
            }
            f32 output = horizontal::sum(acc);
            for (; i < dim_size; ++i) {
                output += src[i];
            }
            Xz[o] = output;
        }
        return;
    }

    for (size_t o = 0; o < outer_size; ++o) {
        f32* dst = Xz + (o * inner_size);

        const f32* src_0 = Xx + (o * dim_size * inner_size);
        size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            storeu(dst + i, loadu(src_0 + i));
        }
        for (; i < inner_size; ++i) {
            dst[i] = src_0[i];
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const f32* src = Xx + ((o * dim_size + d) * inner_size);
            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f out_val = loadu(dst + i);
                const vec8f in_val  = loadu(src + i);
                storeu(dst + i, add(out_val, in_val));
            }
            for (; i < inner_size; ++i) {
                dst[i] += src[i];
            }
        }
    }
}
// output = horizontal::sum(acc);
// output *= invN;
void reduce::mean(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
    sum(Xx, Xz, N);
    Xz[0] /= static_cast<f32>(N);
}

void reduce::mean(const f32 *Xx, f32 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    sum(Xx, Xz, outer_size, dim_size, inner_size);

    const size_t total_output_elements = outer_size * inner_size;
    size_t i = 0;

    const vec8f v_scale = set1(1.0f / static_cast<f32>(dim_size));

    for (; i + 8 <= total_output_elements; i += 8) {
        const vec8f val = loadu(Xz + i);
        storeu(Xz + i, mul(val, v_scale));
    }
    for (; i < total_output_elements; ++i) {
        Xz[i] /= static_cast<f32>(dim_size);
    }
}
// Welford
void reduce::var(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
    mean(Xx, Xz, N);
    const f32 mu = Xz[0];
    const vec8f vmu = set1(mu);

    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        const vec8f diff = sub(loadu(Xx + i), vmu);
        acc = fma::add(diff, diff, acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        const f32 diff = Xx[i] - mu;
        output += diff * diff;
    }
    Xz[0] =  output / static_cast<f32>(N);
}

void reduce::var(const f32 *Xx, f32 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    if (inner_size == 1) {
        for (size_t o = 0; o < outer_size; ++o) {
            const f32* src = Xx + (o * dim_size);

            size_t i = 0;
            vec8f acc_sum = zero();
            for (; i + 8 <= dim_size; i += 8) {
                acc_sum = add(acc_sum, loadu(src + i));
            }
            f32 sum_val = horizontal::sum(acc_sum);
            for (; i < dim_size; ++i) { sum_val += src[i]; }
            const f32 mu = sum_val / static_cast<f32>(dim_size);
            const vec8f vmu = set1(mu);

            i = 0;
            vec8f acc_var = zero();
            for (; i + 8 <= dim_size; i += 8) {
                const vec8f diff = sub(loadu(src + i), vmu);
                acc_var = fma::add(diff, diff, acc_var);
            }
            f32 var_val = horizontal::sum(acc_var);
            for (; i < dim_size; ++i) {
                const f32 diff = src[i] - mu;
                var_val += diff * diff;
            }
            Xz[o] = var_val / static_cast<f32>(dim_size);
        }
        return;
    }

    std::vector<f32> temp_mean(outer_size * inner_size);
    mean(Xx, temp_mean.data(), outer_size, dim_size, inner_size);

    for (size_t o = 0; o < outer_size; ++o) {
        f32* dst = Xz + (o * inner_size);
        const f32* src_0 = Xx + (o * dim_size * inner_size);
        const f32* mu_ptr = temp_mean.data() + (o * inner_size);

        size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            const vec8f vmu = loadu(mu_ptr + i);
            const vec8f diff = sub(loadu(src_0 + i), vmu);
            storeu(dst + i, mul(diff, diff));
        }
        for (; i < inner_size; ++i) {
            const f32 diff = src_0[i] - mu_ptr[i];
            dst[i] = diff * diff;
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const f32* src = Xx + ((o * dim_size + d) * inner_size);
            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f vmu = loadu(mu_ptr + i);
                const vec8f out_val = loadu(dst + i);
                const vec8f diff = sub(loadu(src + i), vmu);
                storeu(dst + i, fma::add(diff, diff, out_val));
            }
            for (; i < inner_size; ++i) {
                const f32 diff = src[i] - mu_ptr[i];
                dst[i] += diff * diff;
            }
        }
    }

    const size_t total_output_elements = outer_size * inner_size;
    size_t i = 0;
    const vec8f v_scale = set1(1.0f / static_cast<f32>(dim_size));
    for (; i + 8 <= total_output_elements; i += 8) {
        storeu(Xz + i, mul(loadu(Xz + i), v_scale));
    }
    for (; i < total_output_elements; ++i) {
        Xz[i] /= static_cast<f32>(dim_size);
    }
}

void reduce::stdv(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
    var(Xx, Xz, N);
    Xz[0] = std::sqrt(Xz[0]);
}

void reduce::stdv(const f32 *Xx, f32 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    var(Xx, Xz, outer_size, dim_size, inner_size);

    const size_t total_output_elements = outer_size * inner_size;
    size_t i = 0;

    for (; i + 8 <= total_output_elements; i += 8) {
        storeu(Xz + i, avx2::sqrt(loadu(Xz + i)));
    }
    for (; i < total_output_elements; ++i) {
        Xz[i] = std::sqrt(Xz[i]);
    }
}

void reduce::min(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
    size_t i = 0;
    vec8f acc = set1(std::numeric_limits<f32>::max());
    for (; i + 8 <= N; i += 8) {
        acc = avx2::min(acc, loadu(Xx + i));
    }
    f32 output = horizontal::min(acc);
    for (; i < N; ++i) {
        output = std::min(output, Xx[i]);
    }
    Xz[0] = output;
}

void reduce::min(const f32 *Xx, f32 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    if (inner_size == 1) {
        for (size_t o = 0; o < outer_size; ++o) {
            const f32* src = Xx + (o * dim_size);
            size_t i = 0;
            vec8f acc = set1(std::numeric_limits<f32>::max());
            for (; i + 8 <= dim_size; i += 8) {
                acc = avx2::min(acc, loadu(src + i));
            }
            f32 output = horizontal::min(acc);
            for (; i < dim_size; ++i) {
                output = std::min(output, src[i]);
            }
            Xz[o] = output;
        }
        return;
    }

    for (size_t o = 0; o < outer_size; ++o) {
        f32* dst = Xz + (o * inner_size);
        const f32* src_0 = Xx + (o * dim_size * inner_size);

        size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            storeu(dst + i, loadu(src_0 + i));
        }
        for (; i < inner_size; ++i) {
            dst[i] = src_0[i];
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const f32* src = Xx + ((o * dim_size + d) * inner_size);
            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f out_val = loadu(dst + i);
                const vec8f in_val  = loadu(src + i);
                storeu(dst + i, avx2::min(out_val, in_val));
            }
            for (; i < inner_size; ++i) {
                dst[i] = std::min(dst[i], src[i]);
            }
        }
    }
}

void reduce::max(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
    size_t i = 0;
    vec8f acc = set1(std::numeric_limits<f32>::lowest());
    for (; i + 8 <= N; i += 8) {
        acc = avx2::max(acc, loadu(Xx + i));
    }
    f32 output = horizontal::max(acc);
    for (; i < N; ++i) {
        output = std::max(output, Xx[i]);
    }
    Xz[0] = output;
}

void reduce::max(const f32 *Xx, f32 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    if (inner_size == 1) {
        for (size_t o = 0; o < outer_size; ++o) {
            const f32* src = Xx + (o * dim_size);
            size_t i = 0;
            vec8f acc = set1(std::numeric_limits<f32>::lowest());
            for (; i + 8 <= dim_size; i += 8) {
                acc = avx2::max(acc, loadu(src + i));
            }
            f32 output = horizontal::max(acc);
            for (; i < dim_size; ++i) {
                output = std::max(output, src[i]);
            }
            Xz[o] = output;
        }
        return;
    }

    for (size_t o = 0; o < outer_size; ++o) {
        f32* dst = Xz + (o * inner_size);
        const f32* src_0 = Xx + (o * dim_size * inner_size);

        size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            storeu(dst + i, loadu(src_0 + i));
        }
        for (; i < inner_size; ++i) {
            dst[i] = src_0[i];
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const f32* src = Xx + ((o * dim_size + d) * inner_size);
            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f out_val = loadu(dst + i);
                const vec8f in_val  = loadu(src + i);
                storeu(dst + i, avx2::max(out_val, in_val));
            }
            for (; i < inner_size; ++i) {
                dst[i] = std::max(dst[i], src[i]);
            }
        }
    }
}

void reduce::norm1(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = add(acc, avx2::abs(loadu(Xx + i)));
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += std::abs(Xx[i]);
    }
    Xz[0] = output;
}

void reduce::norm1(const f32 *Xx, f32 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    if (inner_size == 1) {
        for (size_t o = 0; o < outer_size; ++o) {
            const f32* src = Xx + (o * dim_size);
            size_t i = 0;
            vec8f acc = zero();
            for (; i + 8 <= dim_size; i += 8) {
                acc = add(acc, avx2::abs(loadu(src + i)));
            }
            f32 output = horizontal::sum(acc);
            for (; i < dim_size; ++i) {
                output += std::abs(src[i]);
            }
            Xz[o] = output;
        }
        return;
    }

    for (size_t o = 0; o < outer_size; ++o) {
        f32* dst = Xz + (o * inner_size);
        const f32* src_0 = Xx + (o * dim_size * inner_size);

        size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            storeu(dst + i, avx2::abs(loadu(src_0 + i)));
        }
        for (; i < inner_size; ++i) {
            dst[i] = std::abs(src_0[i]);
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const f32* src = Xx + ((o * dim_size + d) * inner_size);
            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f out_val = loadu(dst + i);
                const vec8f in_val  = avx2::abs(loadu(src + i));
                storeu(dst + i, add(out_val, in_val));
            }
            for (; i < inner_size; ++i) {
                dst[i] += std::abs(src[i]);
            }
        }
    }
}

void reduce::norm2(const f32* __restrict Xx, f32* __restrict Xz, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        const vec8f v = loadu(Xx + i);
        acc = fma::add(v, v, acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += Xx[i] * Xx[i];
    }
    Xz[0] = std::sqrt(output);
}

void reduce::norm2(const f32 *Xx, f32 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    if (inner_size == 1) {
        for (size_t o = 0; o < outer_size; ++o) {
            const f32* src = Xx + (o * dim_size);
            size_t i = 0;
            vec8f acc = zero();
            for (; i + 8 <= dim_size; i += 8) {
                const vec8f v = loadu(src + i);
                acc = fma::add(v, v, acc);
            }
            f32 output = horizontal::sum(acc);
            for (; i < dim_size; ++i) {
                output += src[i] * src[i];
            }
            Xz[o] = std::sqrt(output);
        }
        return;
    }

    for (size_t o = 0; o < outer_size; ++o) {
        f32* dst = Xz + (o * inner_size);
        const f32* src_0 = Xx + (o * dim_size * inner_size);

        size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            const vec8f v = loadu(src_0 + i);
            storeu(dst + i, mul(v, v));
        }
        for (; i < inner_size; ++i) {
            dst[i] = src_0[i] * src_0[i];
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const f32* src = Xx + ((o * dim_size + d) * inner_size);
            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f out_val = loadu(dst + i);
                const vec8f v = loadu(src + i);
                storeu(dst + i, fma::add(v, v, out_val));
            }
            for (; i < inner_size; ++i) {
                dst[i] += src[i] * src[i];
            }
        }
    }

    const size_t total_output_elements = outer_size * inner_size;
    size_t i = 0;
    for (; i + 8 <= total_output_elements; i += 8) {
        storeu(Xz + i, avx2::sqrt(loadu(Xz + i)));
    }
    for (; i < total_output_elements; ++i) {
        Xz[i] = std::sqrt(Xz[i]);
    }
}

void reduce::argmax(const f32* __restrict Xx, i64* __restrict Xz, const size_t N) {
    if (N == 0) return;

    vec8f max_vals = set1(-std::numeric_limits<f32>::infinity());
    vec8i max_indices = _mm256_set1_epi32(-1);

    vec8i curr_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const vec8i idx_step = _mm256_set1_epi32(8);

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f new_vals = loadu(Xx + i);
        const vec8f mask = cmp::gt(new_vals, max_vals);

        max_vals = blendv(max_vals, new_vals, mask);
        max_indices = _mm256_blendv_epi8(max_indices, curr_indices, _mm256_castps_si256(mask));

        curr_indices = _mm256_add_epi32(curr_indices, idx_step);
    }

    alignas(32) f32 values[8];
    alignas(32) i32 indices[8];
    storeu(values, max_vals);
    _mm256_storeu_si256(reinterpret_cast<vec8i *>(indices), max_indices);

    f32 global_max = values[0];
    i64 global_max_idx = indices[0];

    for (size_t j = 1; j < 8; ++j) {
        if (values[j] > global_max) {
            global_max = values[j];
            global_max_idx = indices[j];
        }
    }

    for (; i < N; ++i) {
        if (Xx[i] > global_max || global_max_idx == -1) {
            global_max = Xx[i];
            global_max_idx = static_cast<i64>(i);
        }
    }

    Xz[0] = global_max_idx;
}

void reduce::argmax(const f32 *Xx, i64 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    if (inner_size == 1) {
        for (size_t o = 0; o < outer_size; ++o) {
            argmax(Xx + (o * dim_size), Xz + o, dim_size);
        }
        return;
    }

    const size_t total_output = outer_size * inner_size;
    std::vector<f32> max_vals(total_output);
    std::vector max_idx(total_output, 0);

    for (size_t o = 0; o < outer_size; ++o) {
        const size_t offset = o * inner_size;
        f32* out_val = max_vals.data() + offset;
        int32_t* out_idx = max_idx.data() + offset;

        const f32* src_0 = Xx + (o * dim_size * inner_size);
        size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            storeu(out_val + i, _mm256_loadu_ps(src_0 + i));
            _mm256_storeu_si256(reinterpret_cast<vec8i *>(out_idx + i), _mm256_setzero_si256());
        }
        for (; i < inner_size; ++i) {
            out_val[i] = src_0[i];
            out_idx[i] = 0;
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const f32* src = Xx + ((o * dim_size + d) * inner_size);
            const vec8i vd = _mm256_set1_epi32(static_cast<i32>(d));

            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f old_max = loadu(out_val + i);
                const vec8f new_val = loadu(src + i);

                const vec8f mask = cmp::gt(new_val, old_max);

                storeu(out_val + i, blendv(old_max, new_val, mask));

                const vec8i old_idx = _mm256_loadu_si256(reinterpret_cast<const vec8i *>(out_idx + i));
                _mm256_storeu_si256(reinterpret_cast<vec8i *>(out_idx + i), _mm256_blendv_epi8(old_idx, vd, _mm256_castps_si256(mask)));
            }
            for (; i < inner_size; ++i) {
                if (src[i] > out_val[i]) {
                    out_val[i] = src[i];
                    out_idx[i] = static_cast<i32>(d);
                }
            }
        }
    }

    size_t i = 0;
    for (; i + 4 <= total_output; i += 4) {
        const vec4i low = _mm_loadu_si128(reinterpret_cast<const vec4i *>(max_idx.data() + i));
        const vec8i wide = _mm256_cvtepi32_epi64(low);
        _mm256_storeu_si256(reinterpret_cast<vec8i *>(Xz + i), wide);
    }
    for (; i < total_output; ++i) {
        Xz[i] = static_cast<int64_t>(max_idx[i]);
    }
}

void reduce::argmin(const f32* __restrict Xx, i64* __restrict Xz, const size_t N) {
    if (N == 0) return;

    vec8f min_vals = set1(std::numeric_limits<f32>::max());
    vec8i min_indices = _mm256_set1_epi32(-1);

    vec8i curr_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const vec8i idx_step = _mm256_set1_epi32(8);

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        const vec8f new_vals = loadu(Xx + i);
        const vec8f mask = cmp::lt(new_vals, min_vals);

        min_vals = blendv(min_vals, new_vals, mask);
        min_indices = _mm256_blendv_epi8(min_indices, curr_indices, _mm256_castps_si256(mask));

        curr_indices = _mm256_add_epi32(curr_indices, idx_step);
    }

    alignas(32) f32 values[8];
    alignas(32) i32 indices[8];
    storeu(values, min_vals);
    _mm256_storeu_si256(reinterpret_cast<vec8i *>(indices), min_indices);

    f32 global_min = values[0];
    i64 global_min_idx = indices[0];

    for (size_t j = 1; j < 8; ++j) {
        if (values[j] < global_min) {
            global_min = values[j];
            global_min_idx = indices[j];
        }
    }

    for (; i < N; ++i) {
        if (Xx[i] < global_min || global_min_idx == -1) {
            global_min = Xx[i];
            global_min_idx = static_cast<i64>(i);
        }
    }

    Xz[0] = global_min_idx;
}

void reduce::argmin(const f32 *Xx, i64 *Xz, const size_t outer_size, const size_t dim_size, const size_t inner_size) {
    if (inner_size == 1) {
        for (size_t o = 0; o < outer_size; ++o) {
            argmin(Xx + (o * dim_size), Xz + o, dim_size);
        }
        return;
    }

    const size_t total_output = outer_size * inner_size;
    std::vector<f32> min_vals(total_output);
    std::vector min_idx(total_output, 0);

    for (size_t o = 0; o < outer_size; ++o) {
        const size_t offset = o * inner_size;
        f32* out_val = min_vals.data() + offset;
        int32_t* out_idx = min_idx.data() + offset;

        const f32* src_0 = Xx + (o * dim_size * inner_size);
        size_t i = 0;
        for (; i + 8 <= inner_size; i += 8) {
            _mm256_storeu_ps(out_val + i, _mm256_loadu_ps(src_0 + i));
            _mm256_storeu_si256(reinterpret_cast<vec8i *>(out_idx + i), _mm256_setzero_si256());
        }
        for (; i < inner_size; ++i) {
            out_val[i] = src_0[i];
            out_idx[i] = 0;
        }

        for (size_t d = 1; d < dim_size; ++d) {
            const f32* src = Xx + ((o * dim_size + d) * inner_size);
            const vec8i vd = _mm256_set1_epi32(static_cast<int32_t>(d));

            i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f old_min = loadu(out_val + i);
                const vec8f new_val = loadu(src + i);

                const vec8f mask = cmp::lt(new_val, old_min);

                storeu(out_val + i, blendv(old_min, new_val, mask));

                const __m256i old_idx = _mm256_loadu_si256(reinterpret_cast<const vec8i *>(out_idx + i));
                _mm256_storeu_si256(reinterpret_cast<vec8i *>(out_idx + i), _mm256_blendv_epi8(old_idx, vd, _mm256_castps_si256(mask)));
            }
            for (; i < inner_size; ++i) {
                if (src[i] < out_val[i]) {
                    out_val[i] = src[i];
                    out_idx[i] = static_cast<int32_t>(d);
                }
            }
        }
    }

    size_t i = 0;
    for (; i + 4 <= total_output; i += 4) {
        const vec4i low = _mm_loadu_si128(reinterpret_cast<const vec4i *>(min_idx.data() + i));
        const vec8i wide = _mm256_cvtepi32_epi64(low);
        _mm256_storeu_si256(reinterpret_cast<vec8i *>(Xz + i), wide);
    }
    for (; i < total_output; ++i) {
        Xz[i] = static_cast<int64_t>(min_idx[i]);
    }
}

void reduce::dot(const f32* Xx, const f32* Xy, f32* Xz, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = fma::add(loadu(Xx + i), loadu(Xy + i), acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += Xx[i] * Xy[i];
    }
    Xz[0] = output;
}
//
// Created by muham on 8.05.2026.
//

#include "CortexMind/framework/Engine/AVX2/reduce.hpp"
#include <CortexMind/framework/Engine/AVX2/fma.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Engine/AVX2/horizontal.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

static constexpr size_t COL_TILE = 256;

f32 reduce::sum(const f32 *x, const size_t N) {

    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = add(acc, loadu(x + i));
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += x[i];
    }
    return output;
}

f32 reduce::mean(const f32 *x, const size_t N) {
    return sum(x, N) / static_cast<f32>(N);
}

f32 reduce::var(const f32 *x, const size_t N) {
    const f32 mu = mean(x, N);
    const vec8f vmu = set1(mu);

    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        const vec8f diff = sub(loadu(x + i), vmu);
        acc = fma::add(diff, diff, acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        const f32 diff = x[i] - mu;
        output += diff * diff;
    }
    return output / static_cast<f32>(N);
}

f32 reduce::std(const f32* x, const size_t N) {
    return std::sqrt(var(x, N));
}

f32 reduce::min(const f32* x, const size_t N) {
    size_t i = 0;
    vec8f acc = set1(std::numeric_limits<f32>::max());
    for (; i + 8 <= N; i += 8) {
        acc = avx2::min(acc, loadu(x + i));
    }
    f32 output = horizontal::min(acc);
    for (; i < N; ++i) {
        output = std::min(output, x[i]);
    }
    return output;
}

f32 reduce::max(const f32 *x, const size_t N) {
    size_t i = 0;
    vec8f acc = set1(std::numeric_limits<f32>::lowest());
    for (; i + 8 <= N; i += 8) {
        acc = avx2::max(acc, loadu(x + i));
    }
    f32 output = horizontal::max(acc);
    for (; i < N; ++i) {
        output = std::max(output, x[i]);
    }
    return output;
}

f32 reduce::norm1(const f32 *x, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = add(acc, avx2::abs(loadu(x + i)));
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += std::abs(x[i]);
    }
    return output;
}

f32 reduce::norm2(const f32* x, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        const vec8f v = loadu(x + i);
        acc = fma::add(v, v, acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += x[i] * x[i];
    }
    return std::sqrt(output);
}

f32 reduce::dot(const f32* Xx, const f32* Xy, const size_t N) {
    size_t i = 0;
    vec8f acc = zero();
    for (; i + 8 <= N; i += 8) {
        acc = fma::add(loadu(Xx + i), loadu(Xy + i), acc);
    }
    f32 output = horizontal::sum(acc);
    for (; i < N; ++i) {
        output += Xx[i] * Xy[i];
    }
    return output;
}

bool reduce::equal(const f32 *Xx, const f32 *Xy, const size_t N) {
    size_t i = 0;
    //const vec8f eps = set1(1e-6f);
    vec8f acc = zero();

    for (; i + 8 <= N; i += 8) {
        const vec8f diff = avx2::abs(sub(loadu(Xx + i), loadu(Xy + i)));
        acc = avx2::max(acc, diff);
    }

    f32 max_diff = horizontal::max(acc);
    for (; i < N; ++i) {
        max_diff = std::max(max_diff, std::abs(Xx[i] - Xy[i]));
    }

    return max_diff < 1e-6f;
}

void reduce::sum_last_dim(const f32 *src, f32 *dst, const size_t rows, const size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        dst[r] = sum(src + r * cols, cols);
    }
}

void reduce::sum_first_dim(const f32 *src, f32 *dst, const size_t rows, const size_t cols) {
    size_t col = 0;

    for (; col + COL_TILE <= cols; col += COL_TILE) {
        f32* out = dst + col;

        for (size_t r = 0; r < rows; ++r) {
            const f32* row = src + r * cols + col;
            size_t i = 0;
            for (; i + 8 <= COL_TILE; i += 8) {
                storeu(out + i, add(loadu(out + i), loadu(row + i)));
            }
            for (; i < COL_TILE; ++i) {
                out[i] += row[i];
            }
        }
    }

    const size_t tail = cols - col;
    if (tail == 0) return;

    for (size_t r = 0; r < rows; ++r) {
        const f32* row = src + r * cols + col;
        f32*       out = dst + col;
        size_t i = 0;
        for (; i + 8 <= tail; i += 8) {
            storeu(out + i, add(loadu(out + i), loadu(row + i)));
        }
        for (; i < tail; ++i) {
            out[i] += row[i];
        }
    }
}

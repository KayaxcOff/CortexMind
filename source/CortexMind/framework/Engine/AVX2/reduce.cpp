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

/**
 * @brief Verilen boyutlar boyunca toplam hesapla
 * Yardımcı fonksiyon: Tensor'un belirli boyutları boyunca toplam yapar
 */
static void sum_dim_internal(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims,
    size_t total_elements
) {
    // Tüm elemanlara ulaş
    for (size_t idx = 0; idx < total_elements; ++idx) {
        // Doğrusal indeksi çok boyutlu koordinatlara dönüştür
        size_t oz = 0;
        size_t temp_idx = idx;

        for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
            const size_t coord = temp_idx % static_cast<size_t>(shape[d]);
            temp_idx /= static_cast<size_t>(shape[d]);

            // Bu boyut küçültülüyor mu?
            bool is_reduced = false;
            for (size_t r = 0; r < num_reduce_dims; ++r) {
                if (reduce_dims[r] == d) {
                    is_reduced = true;
                    break;
                }
            }

            const size_t out_coord = is_reduced ? 0 : coord;
            oz += out_coord * static_cast<size_t>(strides[d]);
        }

        output[oz] += x[idx];
    }
}

void reduce::mean_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    // Toplam sayısını hesapla
    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    // Çıkışı sıfırla
    for (size_t d = 0; d < ndim; ++d) {
        size_t out_size = 1;
        for (size_t dd = 0; dd < ndim; ++dd) {
            bool is_reduced = false;
            for (size_t r = 0; r < num_reduce_dims; ++r) {
                if (reduce_dims[r] == static_cast<i64>(dd)) {
                    is_reduced = true;
                    break;
                }
            }
            if (!is_reduced) {
                out_size *= static_cast<size_t>(shape[dd]);
            }
        }
        if (d == 0) {
            for (size_t i = 0; i < out_size; ++i) {
                output[i] = 0.0f;
            }
        }
    }

    // Toplamı hesapla
    sum_dim_internal(x, output, shape, strides, reduce_dims, ndim, num_reduce_dims, total_elements);

    // Küçültülen boyutların çarpımını hesapla (normalize için)
    size_t reduce_count = 1;
    for (size_t r = 0; r < num_reduce_dims; ++r) {
        reduce_count *= static_cast<size_t>(shape[reduce_dims[r]]);
    }

    // Normalizasyon: Sonuçları küçültülen eleman sayısına böl
    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }

    const f32 divisor = static_cast<f32>(reduce_count);
    for (size_t i = 0; i < out_size; ++i) {
        output[i] /= divisor;
    }
}

void reduce::var_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    // Önce ortalamayı hesapla
    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }

    std::vector<f32> mean_values(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        mean_values[i] = 0.0f;
    }

    // Ortalamayı hesapla
    mean_dim(x, mean_values.data(), shape, strides, reduce_dims, ndim, num_reduce_dims);

    // Varyans hesapla: (x - mean)^2
    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    for (size_t i = 0; i < out_size; ++i) {
        output[i] = 0.0f;
    }

    // Varyansı hesapla: ∑((x - mean)^2) / count
    for (size_t idx = 0; idx < total_elements; ++idx) {
        size_t oz = 0;
        size_t temp_idx = idx;

        for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
            const size_t coord = temp_idx % static_cast<size_t>(shape[d]);
            temp_idx /= static_cast<size_t>(shape[d]);

            bool is_reduced = false;
            for (size_t r = 0; r < num_reduce_dims; ++r) {
                if (reduce_dims[r] == d) {
                    is_reduced = true;
                    break;
                }
            }

            const size_t out_coord = is_reduced ? 0 : coord;
            oz += out_coord * static_cast<size_t>(strides[d]);
        }

        const f32 diff = x[idx] - mean_values[oz];
        output[oz] += diff * diff;
    }

    // Normalizasyon
    size_t reduce_count = 1;
    for (size_t r = 0; r < num_reduce_dims; ++r) {
        reduce_count *= static_cast<size_t>(shape[reduce_dims[r]]);
    }

    const f32 divisor = static_cast<f32>(reduce_count);
    for (size_t i = 0; i < out_size; ++i) {
        output[i] /= divisor;
    }
}

void reduce::std_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }

    // Varyansı hesapla
    var_dim(x, output, shape, strides, reduce_dims, ndim, num_reduce_dims);

    // Karekök al
    for (size_t i = 0; i < out_size; ++i) {
        output[i] = std::sqrt(output[i]);
    }
}

void reduce::min_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }

    // Çıkışı max değere başlat
    for (size_t i = 0; i < out_size; ++i) {
        output[i] = std::numeric_limits<f32>::max();
    }

    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    // Minimum değeri bul
    for (size_t idx = 0; idx < total_elements; ++idx) {
        size_t oz = 0;
        size_t temp_idx = idx;

        for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
            const size_t coord = temp_idx % static_cast<size_t>(shape[d]);
            temp_idx /= static_cast<size_t>(shape[d]);

            bool is_reduced = false;
            for (size_t r = 0; r < num_reduce_dims; ++r) {
                if (reduce_dims[r] == d) {
                    is_reduced = true;
                    break;
                }
            }

            const size_t out_coord = is_reduced ? 0 : coord;
            oz += out_coord * static_cast<size_t>(strides[d]);
        }

        output[oz] = std::min(output[oz], x[idx]);
    }
}

void reduce::max_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }

    // Çıkışı min değere başlat
    for (size_t i = 0; i < out_size; ++i) {
        output[i] = std::numeric_limits<f32>::lowest();
    }

    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    // Maksimum değeri bul
    for (size_t idx = 0; idx < total_elements; ++idx) {
        size_t oz = 0;
        size_t temp_idx = idx;

        for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
            const size_t coord = temp_idx % static_cast<size_t>(shape[d]);
            temp_idx /= static_cast<size_t>(shape[d]);

            bool is_reduced = false;
            for (size_t r = 0; r < num_reduce_dims; ++r) {
                if (reduce_dims[r] == d) {
                    is_reduced = true;
                    break;
                }
            }

            const size_t out_coord = is_reduced ? 0 : coord;
            oz += out_coord * static_cast<size_t>(strides[d]);
        }

        output[oz] = std::max(output[oz], x[idx]);
    }
}
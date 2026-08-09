//
// Created by muham on 9.08.2026.
//

#include "CortexMind/framework/Engine/AVX2/broadcast.hpp"
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <vector>

using namespace cortex::_fw::avx2;
using namespace cortex::_fw;

namespace {
    template<typename OpVec, typename OpScalar>
    void general_kernel(const float* __restrict x, const float* __restrict y, float* __restrict z, const BroadcastInfo& info, OpVec op_vec, OpScalar op_scalar) {
        const bool contiguous = (info.stride_x[info.ndim - 1] == 1 && info.stride_y[info.ndim - 1] == 1 && info.stride_z[info.ndim - 1] == 1);

        if (!contiguous) {
            std::size_t total = 1;
            for (std::int32_t d = 0; d < info.ndim; ++d) total *= info.shape[d];

            for (std::size_t i = 0; i < total; ++i) {
                size_t ox = 0, oy = 0, oz = 0, idx = i;
                for (std::int32_t  d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = idx % info.shape[d];
                    ox += coord * info.stride_x[d];
                    oy += coord * info.stride_y[d];
                    oz += coord * info.stride_z[d];
                    idx /= info.shape[d];
                }
                z[oz] = op_scalar(x[ox], y[oy]);
            }
            return;
        }

        const std::size_t inner_size = info.shape[info.ndim - 1];
        std::vector<std::size_t> indices(info.ndim, 0);

        while (true) {
            std::size_t ox = 0, oy = 0, oz = 0;
            for (std::int32_t d = 0; d < info.ndim - 1; ++d) {
                ox += indices[d] * info.stride_x[d];
                oy += indices[d] * info.stride_y[d];
                oz += indices[d] * info.stride_z[d];
            }

            std::size_t i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f xv = loadu(x + ox + i);
                const vec8f yv = loadu(y + oy + i);
                storeu(z + oz + i, op_vec(xv, yv));
            }

            for (; i < inner_size; ++i) {
                z[oz + i] = op_scalar(x[ox + i], y[oy + i]);
            }

            int d = info.ndim - 2;
            while (d >= 0) {
                indices[d]++;
                if (indices[d] < info.shape[d]) {
                    break;
                }
                indices[d] = 0;
                d--;
            }

            if (d < 0) {
                break;
            }
        }
    }

    template<typename OpVec, typename OpScalar>
    void general_kernel_inplace(float* x, const float* __restrict y, const BroadcastInfo& info, OpVec op_vec, OpScalar op_scalar) {
        const bool contiguous = (info.stride_x[info.ndim - 1] == 1 && info.stride_y[info.ndim - 1] == 1);

        if (!contiguous) {
            std::size_t total = 1;
            for (std::int32_t d = 0; d < info.ndim; ++d) total *= info.shape[d];

            for (std::size_t i = 0; i < total; ++i) {
                size_t ox = 0, oy = 0, idx = i;
                for (std::int32_t  d = info.ndim - 1; d >= 0; --d) {
                    const size_t coord = idx % info.shape[d];
                    ox += coord * info.stride_x[d];
                    oy += coord * info.stride_y[d];
                    idx /= info.shape[d];
                }
                x[ox] = op_scalar(x[ox], y[oy]);
            }
            return;
        }

        const std::size_t inner_size = info.shape[info.ndim - 1];
        std::vector<std::size_t> indices(info.ndim, 0);

        while (true) {
            std::size_t ox = 0, oy = 0;
            for (std::int32_t d = 0; d < info.ndim - 1; ++d) {
                ox += indices[d] * info.stride_x[d];
                oy += indices[d] * info.stride_y[d];
            }

            std::size_t i = 0;
            for (; i + 8 <= inner_size; i += 8) {
                const vec8f xv = loadu(x + ox + i);
                const vec8f yv = loadu(y + oy + i);
                storeu(x + ox + i, op_vec(xv, yv));
            }

            for (; i < inner_size; ++i) {
                x[ox + i] = op_scalar(x[ox + i], y[oy + i]);
            }

            int d = info.ndim - 2;
            while (d >= 0) {
                indices[d]++;
                if (indices[d] < info.shape[d]) {
                    break;
                }
                indices[d] = 0;
                d--;
            }

            if (d < 0) {
                break;
            }
        }
    }
} //unnamed namespace

void Broadcast::row::add(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::add(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] + Xy[j];
        }
    }
}

void Broadcast::row::sub(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::sub(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] - Xy[j];
        }
    }
}

void Broadcast::row::mul(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::mul(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] * Xy[j];
        }
    }
}

void Broadcast::row::div(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::mul(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] * Xy[j];
        }
    }
}

void Broadcast::row::add(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::add(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] + Xy[j];
        }
    }
}

void Broadcast::row::sub(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::sub(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] - Xy[j];
        }
    }
}

void Broadcast::row::mul(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::mul(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] * Xy[j];
        }
    }
}

void Broadcast::row::div(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::sub(loadu(Xx + i * N + j), loadu(Xy + j)));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] / Xy[j];
        }
    }
}

void Broadcast::col::add(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::add(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] + val;
        }
    }
}

void Broadcast::col::sub(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::sub(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] - val;
        }
    }
}

void Broadcast::col::mul(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::mul(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] * val;
        }
    }
}

void Broadcast::col::div(const float *Xx, const float *Xy, float *Xz, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xz + i * N + j, avx2::div(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xz + i * N)[j] = (Xx + i * N)[j] / val;
        }
    }
}

void Broadcast::col::add(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::add(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] + val;
        }
    }
}

void Broadcast::col::sub(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::sub(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] - val;
        }
    }
}

void Broadcast::col::mul(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::mul(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] * val;
        }
    }
}

void Broadcast::col::div(float *Xx, const float *Xy, const std::size_t M, const std::size_t N) {
    for (std::size_t i = 0; i < M; ++i) {
        const float val = Xy[i];
        const auto vval = set1(val);
        std::size_t j = 0;
        for (; j + 8 <= N; j += 8) {
            storeu(Xx + i * N + j, avx2::div(loadu(Xx + i * N + j), vval));
        }
        for (; j < N; ++j) {
            (Xx + i * N)[j] = (Xx + i * N)[j] / val;
        }
    }
}

void Broadcast::general::add(const float *Xx, const float *Xy, float *Xz, const BroadcastInfo &info) {
    general_kernel(Xx, Xy, Xz, info, [](const vec8f x, const vec8f y){return avx2::add(x, y);}, [](const float x, const float y){return x + y;});
}

void Broadcast::general::sub(const float *Xx, const float *Xy, float *Xz, const BroadcastInfo &info) {
    general_kernel(Xx, Xy, Xz, info, [](const vec8f x, const vec8f y){return avx2::sub(x, y);}, [](const float x, const float y){return x - y;});
}

void Broadcast::general::mul(const float *Xx, const float *Xy, float *Xz, const BroadcastInfo &info) {
    general_kernel(Xx, Xy, Xz, info, [](const vec8f x, const vec8f y){return avx2::mul(x, y);}, [](const float x, const float y){return x * y;});
}

void Broadcast::general::div(const float *Xx, const float *Xy, float *Xz, const BroadcastInfo &info) {
    general_kernel(Xx, Xy, Xz, info, [](const vec8f x, const vec8f y){return avx2::div(x, y);}, [](const float x, const float y){return x / y;});
}

void Broadcast::general::add(float *Xx, const float *Xy, const BroadcastInfo &info) {
    general_kernel_inplace(Xx, Xy, info, [](const vec8f x, const vec8f y){return avx2::add(x, y);}, [](const float x, const float y){return x + y;});
}

void Broadcast::general::sub(float *Xx, const float *Xy, const BroadcastInfo &info) {
    general_kernel_inplace(Xx, Xy, info, [](const vec8f x, const vec8f y){return avx2::sub(x, y);}, [](const float x, const float y){return x - y;});
}

void Broadcast::general::mul(float *Xx, const float *Xy, const BroadcastInfo &info) {
    general_kernel_inplace(Xx, Xy, info, [](const vec8f x, const vec8f y){return avx2::mul(x, y);}, [](const float x, const float y){return x * y;});
}

void Broadcast::general::div(float *Xx, const float *Xy, const BroadcastInfo &info) {
    general_kernel_inplace(Xx, Xy, info, [](const vec8f x, const vec8f y){return avx2::div(x, y);}, [](const float x, const float y){return x / y;});
}
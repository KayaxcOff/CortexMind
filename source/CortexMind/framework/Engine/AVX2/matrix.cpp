//
// Created by muham on 8.05.2026.
//

#include "CortexMind/framework/Engine/AVX2/matrix.hpp"
#include <CortexMind/framework/Engine/AVX2/fma.hpp>
#include <CortexMind/framework/Engine/AVX2/functions.hpp>
#include <CortexMind/framework/Engine/AVX2/partial.hpp>
#include <algorithm>
#include <xutility>

using namespace cortex::_fw::avx2;

void matrix_t::add(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::add(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] + Xy[i];
    }
}

void matrix_t::sub(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::sub(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] - Xy[i];
    }
}

void matrix_t::mul(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::mul(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] * Xy[i];
    }
}

void matrix_t::div(const f32 *Xx, const f32 *Xy, f32 *Xz, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xz + i, avx2::div(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xz[i] = Xx[i] / Xy[i];
    }
}

void matrix_t::matmul(const f32* Xx, const f32* Xy, f32* Xz, const size_t xN, const size_t yN, const size_t zN) {
    std::fill_n(Xz, xN * zN, 0.0f);

    constexpr size_t NC = 256;

    for (size_t jc = 0; jc < zN; jc += NC) {
        constexpr size_t KC = 256;
        const size_t jc_end = std::min(jc + NC, zN);

        for (size_t kc = 0; kc < yN; kc += KC) {
            constexpr size_t MC = 96;
            const size_t kc_end = std::min(kc + KC, yN);

            for (size_t ic = 0; ic < xN; ic += MC) {
                constexpr size_t MR = 8;
                const size_t ic_end = std::min(ic + MC, xN);

                for (size_t i = ic; i < ic_end; i += MR) {
                    constexpr size_t NR = 8;
                    const size_t ib = std::min(MR, ic_end - i);

                    for (size_t j = jc; j < jc_end; j += NR) {
                        const size_t jb = std::min(NR, jc_end - j);

                        vec8f acc[MR];
                        for (size_t r = 0; r < ib; ++r) {
                            acc[r] = zero();
                        }

                        if (ib == MR && jb == NR) {
                            for (size_t k = kc; k < kc_end; ++k) {
                                const vec8f b_vec = loadu(Xy + k * zN + j);

                                acc[0] = fma::add(set1(Xx[(i + 0) * yN + k]), b_vec, acc[0]);
                                acc[1] = fma::add(set1(Xx[(i + 1) * yN + k]), b_vec, acc[1]);
                                acc[2] = fma::add(set1(Xx[(i + 2) * yN + k]), b_vec, acc[2]);
                                acc[3] = fma::add(set1(Xx[(i + 3) * yN + k]), b_vec, acc[3]);
                                acc[4] = fma::add(set1(Xx[(i + 4) * yN + k]), b_vec, acc[4]);
                                acc[5] = fma::add(set1(Xx[(i + 5) * yN + k]), b_vec, acc[5]);
                                acc[6] = fma::add(set1(Xx[(i + 6) * yN + k]), b_vec, acc[6]);
                                acc[7] = fma::add(set1(Xx[(i + 7) * yN + k]), b_vec, acc[7]);
                            }

                            for (size_t r = 0; r < MR; ++r) {
                                const vec8f prev = loadu(Xz + (i + r) * zN + j);
                                storeu(Xz + (i + r) * zN + j, avx2::add(prev, acc[r]));
                            }
                        } else {
                            for (size_t k = kc; k < kc_end; ++k) {
                                const vec8f b_vec = (jb == NR) ? loadu(Xy + k * zN + j) : partial::load(Xy + k * zN + j, jb);

                                for (size_t r = 0; r < ib; ++r) {
                                    acc[r] = fma::add(set1(Xx[(i + r) * yN + k]), b_vec, acc[r]);
                                }
                            }

                            for (size_t r = 0; r < ib; ++r) {
                                if (jb == NR) {
                                    const vec8f prev = loadu(Xz + (i + r) * zN + j);
                                    storeu(Xz + (i + r) * zN + j, avx2::add(prev, acc[r]));
                                } else {
                                    const vec8f prev = partial::load(Xz + (i + r) * zN + j, jb);
                                    partial::store(Xz + (i + r) * zN + j, avx2::add(prev, acc[r]), jb);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void matrix_t::add(f32 *Xx, const f32 *Xy, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::add(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] + Xy[i];
    }
}

void matrix_t::sub(f32 *Xx, const f32 *Xy, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::sub(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] - Xy[i];
    }
}

void matrix_t::mul(f32 *Xx, const f32 *Xy, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::mul(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] * Xy[i];
    }
}

void matrix_t::div(f32 *Xx, const f32 *Xy, const size_t N) {
    if (N <= 0) {
        return;
    }

    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
        storeu(Xx + i, avx2::div(loadu(Xx + i), loadu(Xy + i)));
    }
    for (; i < N; ++i) {
        Xx[i] = Xx[i] / Xy[i];
    }
}

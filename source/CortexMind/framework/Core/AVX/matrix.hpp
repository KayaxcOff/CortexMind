//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_MATRIX_HPP
#define CORTEXMIND_MATRIX_HPP

#include <CortexMind/framework/Core/AVX/funcs.hpp>

namespace cortex::_fw::avx2 {
    typedef struct TensorCalculate {
        static void add(const float* a, const float* b, float* result, const size_t idx) {
            size_t i = 0;
            for (; i + 8 <= idx; i += 8) {
                const reg va = load(a + i);
                const reg vb = load(b + i);
                store(result + i, avx2::add(va, vb));
            }
            if (i < idx) {
                const reg va = load_partial(a + i, idx - i);
                const reg vb = load_partial(b + i, idx - i);
                store_partial(result + i, avx2::add(va, vb), idx - i);
            }
        }

        static void sub(const float* a, const float* b, float* result, const size_t idx) {
            size_t i = 0;
            for (; i + 8 <= idx; i += 8) {
                const reg va = load(a + i);
                const reg vb = load(b + i);
                store(result + i, avx2::sub(va, vb));
            }
            if (i < idx) {
                const reg va = load_partial(a + i, idx - i);
                const reg vb = load_partial(b + i, idx - i);
                store_partial(result + i, avx2::sub(va, vb), idx - i);
            }
        }

        static void mul(const float* a, const float* b, float* result, const size_t idx) {
            size_t i = 0;
            for (; i + 8 <= idx; i += 8) {
                const reg va = load(a + i);
                const reg vb = load(b + i);
                store(result + i, avx2::mul(va, vb));
            }
            if (i < idx) {
                const reg va = load_partial(a + i, idx - i);
                const reg vb = load_partial(b + i, idx - i);
                store_partial(result + i, avx2::mul(va, vb), idx - i);
            }
        }

        static void div(const float* a, const float* b, float* result, const size_t idx) {
            size_t i = 0;
            for (; i + 8 <= idx; i += 8) {
                const reg va = load(a + i);
                const reg vb = load(b + i);
                store(result + i, avx2::div(va, vb));
            }
            if (i < idx) {
                const reg va = load_partial(a + i, idx - i);
                const reg vb = load_partial(b + i, idx - i);
                store_partial(result + i, avx2::div(va, vb), idx - i);
            }
        }

        static void fma(const float* a, const float* b, float* result, const size_t idx) {
            size_t i = 0;

            for (constexpr size_t block = 8; i + block <= idx; i += block) {
                const reg va = load(a + i);
                const reg vb = load(b + i);
                const reg vr = load(result + i);
                store(result + i, avx2::fma(va, vb, vr));
            }
            if (i < idx) {
                const int rem = static_cast<int>(idx - i);
                const regi mask = _mm256_setr_epi32(
                    rem > 0 ? -1 : 0,
                    rem > 1 ? -1 : 0,
                    rem > 2 ? -1 : 0,
                    rem > 3 ? -1 : 0,
                    rem > 4 ? -1 : 0,
                    rem > 5 ? -1 : 0,
                    rem > 6 ? -1 : 0,
                    rem > 7 ? -1 : 0
                );

                const reg va = _mm256_maskload_ps(a + i, mask);
                const reg vb = _mm256_maskload_ps(b + i, mask);
                const reg vc = _mm256_maskload_ps(result + i, mask);
                const reg vr = avx2::fma(va, vb, vc);
                _mm256_maskstore_ps(result + i, mask, vr);
            }
        }

        static void matmul(const float* a, const float* b, float* result, const size_t M, const size_t N, const size_t K) {
            constexpr size_t block = 8;

            for (size_t i = 0; i < M; i += block) {
                const size_t im = std::min(block, M - i);

                for (size_t j = 0; j < N; j += block) {
                    const size_t jm = std::min(block, N - j);

                    reg acc[block];

                    for (size_t ii = 0; ii < im; ++ii) {
                        acc[ii] = zero();
                    }

                    for (size_t k = 0; k < K; ++k) {
                        reg vec;
                        if (jm == block)
                            vec = load(b + k * N + j);
                        else
                            vec = load_partial(b + k * N + j, jm);

                        for (size_t ii = 0; ii < im; ++ii) {
                            const reg a_val = broadcast(a[(i + ii) * K + k]);
                            acc[ii] = avx2::fma(a_val, vec, acc[ii]);
                        }
                    }

                    for (size_t ii = 0; ii < im; ++ii) {
                        if (jm == block)
                            store(result + (i + ii) * N + j, acc[ii]);
                        else
                            store_partial(result + (i + ii) * N + j, acc[ii], jm);
                    }
                }
            }
        }
    } matrix_t;
}

#endif //CORTEXMIND_MATRIX_HPP
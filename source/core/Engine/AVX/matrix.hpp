//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP

#include <core/Engine/AVX/ops.hpp>
#include <algorithm>

namespace cortex::_fw::avx2 {
    typedef struct TensorOperations {

        /// @brief Adds two float arrays element-wise using AVX2
        /// @param a Pointer to the first input array
        /// @param b Pointer to the second input array
        /// @param c Pointer to the output array where results will be stored
        /// @param idx Number of elements in the arrays
        ///
        /// @details
        /// The function processes 8 elements at a time using AVX2 256-bit vectors for speed.
        /// Any remaining elements (if idx is not a multiple of 8) are processed using a scalar loop.
        static void add(const float* a, const float* b, float* c, const size_t idx) {
            size_t i = 0;
            for (; i + 8 < idx; i += 8) {
                const vec8f va = load_u(a);
                const vec8f vb = load_u(b);
                vec8f vc = avx2::add(va, vb);
                store_u(c, vc);
            }
            for (; i < idx; ++i) {
                c[i] = a[i] + b[i];
            }
        }

        /// @brief Subtracts two float arrays element-wise using AVX2
        static void sub(const float* a, const float* b, float* c, const size_t idx) {
            size_t i = 0;
            for (; i + 8 < idx; i += 8) {
                const vec8f va = load_u(a);
                const vec8f vb = load_u(b);
                vec8f vc = avx2::sub(va, vb);
                store_u(c, vc);
            }
            for (; i < idx; ++i) {
                c[i] = a[i] - b[i];
            }
        }

        /// @brief Multiplies two float arrays element-wise using AVX2
        static void mul(const float* a, const float* b, float* c, const size_t idx) {
            size_t i = 0;
            for (; i + 8 < idx; i += 8) {
                const vec8f va = load_u(a);
                const vec8f vb = load_u(b);
                vec8f vc = avx2::mul(va, vb);
                store_u(c, vc);
            }
            for (; i < idx; ++i) {
                c[i] = a[i] * b[i];
            }
        }

        /// @brief Divides two float arrays element-wise using AVX2
        static void div(const float* a, const float* b, float* c, const size_t idx) {
            size_t i = 0;
            for (; i + 8 < idx; i += 8) {
                const vec8f va = load_u(a);
                const vec8f vb = load_u(b);
                vec8f vc = avx2::div(va, vb);
                store_u(c, vc);
            }
            for (; i < idx; ++i) {
                c[i] = a[i] / b[i];
            }
        }

        /// @brief Performs element-wise fused multiply-add (FMA) on three float arrays using AVX2
        /// @param a Pointer to the first input array (multiplicand)
        /// @param b Pointer to the second input array (multiplier)
        /// @param c Pointer to the third input array (addend)
        /// @param d Pointer to the output array where results will be stored
        /// @param idx Number of elements in the arrays
        ///
        /// @details
        /// The function computes `d[i] = a[i] * b[i] + c[i]` for each element.
        /// It uses AVX2 256-bit vectors to process 8 elements at a time for speed.
        /// Any remaining elements (if idx is not a multiple of 8) are handled with a scalar loop.
        static void fma(const float* a, const float* b, const float* c, float* d, const size_t idx) {
            size_t i = 0;
            for (; i + 8 < idx; i += 8) {
                const vec8f va = load_u(a + i);
                const vec8f vb = load_u(b + i);
                const vec8f vc = load_u(c + i);
                vec8f vd = avx2::fma(va, vb, vc);
                store_u(d + i, vd);
            }
            for (; i < idx; ++i) {
                d[i] = a[i] * b[i] + c[i];
            }
        }

            /// @brief Performs block-based matrix multiplication using AVX2 FMA
    /// @param a Pointer to input matrix A (size M x K)
    /// @param b Pointer to input matrix B (size K x N)
    /// @param c Pointer to output matrix C (size M x N)
    /// @param aIdx Number of rows in matrix A
    /// @param bIdx Number of columns in matrix B
    /// @param cIdx Number of columns in matrix C (same as bIdx)
    ///
    /// @details
    /// The function performs matrix multiplication C = A * B using a blocked approach
    /// to leverage AVX2 256-bit vectors for speed.
    /// Each block is of size 8x8 (blockSize).
    /// For each block, elements are broadcasted from A, multiplied with B block vectors
    /// using FMA, and accumulated into temporary AVX vectors.
    /// Partial blocks at matrix edges are handled using `load_partial` and `store_partial`.
    static void matmul(const float* a, const float* b, float* c, const size_t aIdx, const size_t bIdx, const size_t cIdx) {
        constexpr size_t blockSize = 8;

        for (size_t i = 0; i < aIdx; i += blockSize) {
            const size_t im = std::min(aIdx - i, blockSize);

            for (size_t j = 0; j < bIdx; j += blockSize) {
                const size_t jm = std::min(bIdx - j, blockSize);

                vec8f va[blockSize];

                for (size_t k = 0; k < im; ++k) va[k] = zero();

                for (size_t k = 0; k < cIdx; k += blockSize) {
                    vec8f vb;

                    if (jm == blockSize) vb = load_u(b + k * cIdx + j);
                    else vb = load_partial(b + k * bIdx + j, jm);

                    for (size_t l = 0; l < im; ++l) {
                        const vec8f a_elem = broadcast(a[(i + l) * cIdx + k]);
                        va[l] = avx2::fma(a_elem, vb, va[l]);
                    }
                }

                for (size_t k = 0; k < im; ++k) {
                    if (jm == blockSize) store_u(c + (i + k) * cIdx + j, va[k]);
                    else store_partial(c + (i + k) * cIdx + j, va[k], jm);
                }
            }
        }
    }

    } matrix_t;
} // namespace cortex::_fw::avx2

#endif //CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP
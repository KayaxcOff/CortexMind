#ifndef CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP

#include <CortexMind/core/Engine/AVX/ops.hpp>
#include <cstddef>
#include <algorithm>

namespace cortex::_fw::avx2 {
	// @brief Structure encapsulating tensor operations using AVX2
	typedef struct TensorOperations {
		// @brief Element-wise addition of two float arrays using AVX2
		// @param a Pointer to the first input array
		// @param b Pointer to the second input array
		// @param result Pointer to the output array where the result will be stored
		// @param size Number of elements in the input arrays
		static inline void add(const float* a, const float* b, float* result, size_t size) {
			size_t i = 0;
			for (; i + 8 <= size; i += 8) {
				const vec8f va = loadu(a + i);
				const vec8f vb = loadu(b + i);
				const vec8f vr = avx2::add(va, vb);
				storeu(result + i, vr);
			}
			for (; i < size; ++i) {
				result[i] = a[i] + b[i];
			}
		}

		// @brief Element-wise subtraction of two float arrays using AVX2
		// @param a Pointer to the first input array
		// @param b Pointer to the second input array
		// @param result Pointer to the output array where the result will be stored
		// @param size Number of elements in the input arrays
		static inline void sub(const float* a, const float* b, float* result, size_t size) {
			size_t i = 0;
			for (; i + 8 <= size; i += 8) {
				const vec8f va = loadu(a + i);
				const vec8f vb = loadu(b + i);
				const vec8f vr = avx2::sub(va, vb);
				storeu(result + i, vr);
			}
			for (; i < size; ++i) {
				result[i] = a[i] - b[i];
			}
		}

		// @brief Element-wise multiplication of two float arrays using AVX2
		// @param a Pointer to the first input array
		// @param b Pointer to the second input array
		// @param result Pointer to the output array where the result will be stored
		// @param size Number of elements in the input arrays
		static inline void mul(const float* a, const float* b, float* result, size_t size) {
			size_t i = 0;
			for (; i + 8 <= size; i += 8) {
				const vec8f va = loadu(a + i);
				const vec8f vb = loadu(b + i);
				const vec8f vr = avx2::mul(va, vb);
				storeu(result + i, vr);
			}
			for (; i < size; ++i) {
				result[i] = a[i] * b[i];
			}
		}

		// @brief Element-wise division of two float arrays using AVX2
		// @param a Pointer to the first input array
		// @param b Pointer to the second input array
		// @param result Pointer to the output array where the result will be stored
		// @param size Number of elements in the input arrays
		static inline void div(const float* a, const float* b, float* result, size_t size) {
			size_t i = 0;
			for (; i + 8 <= size; i += 8) {
				const vec8f va = loadu(a + i);
				const vec8f vb = loadu(b + i);
				const vec8f vr = avx2::div(va, vb);
				storeu(result + i, vr);
			}
			for (; i < size; ++i) {
				result[i] = a[i] / b[i];
			}
		}

		// @brief Element-wise fused multiply-add of three float arrays using AVX2
		// @param a Pointer to the first input array
		// @param b Pointer to the second input array
		// @param c Pointer to the third input array
		// @param result Pointer to the output array where the result will be stored
		// @param size Number of elements in the input arrays
		static inline void fma(const float* a, const float* b, const float* c, float* result, const size_t size) {
			size_t i = 0;
			for (; i + 8 <= size; i += 8) {
				const vec8f va = loadu(a + i);
				const vec8f vb = loadu(b + i);
				const vec8f vc = loadu(c + i);
				const vec8f vr = avx2::fmadd(va, vb, vc);
				storeu(result + i, vr);
			}
			for (; i < size; ++i) {
				result[i] = a[i] * b[i] + c[i];
			}
		}

		// @brief Matrix multiplication of two float matrices using AVX2
		// @param A Pointer to the first input matrix (size M x K)
		// @param B Pointer to the second input matrix (size K x N)
		// @param C Pointer to the output matrix where the result will be stored (size M x N)
		// @param M Number of rows in matrix A and C
		// @param N Number of columns in matrix B and C
		// @param K Number of columns in matrix A and rows in matrix B
		static inline void matmul(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
			constexpr size_t blockSize = 8;

			for(size_t i = 0; i < M; i += blockSize) {
				const size_t im = std::min(blockSize, M - i);

				for (size_t j = 0; j < N; j += blockSize) {
					const size_t jn = std::min(blockSize, N - j);

					vec8f acc[blockSize];

					for (size_t k = 0; k < im; ++k) {
						acc[k] = zero();
					}

					for(size_t k = 0; k < K; ++k) {
						vec8f vec;

						if (jn == blockSize) vec = load(B + k * N + j);
						else vec = partial_load(B + k * N + j, jn);

						for (size_t l = 0; l < im; ++l) {
							const vec8f a_elem = _mm256_set1_ps(A[(i + l) * K + k]);
							acc[l] = fma(a_elem, vec, acc[l]);
						}
					}

					for (size_t k = 0; k < im; ++k) {
						if (jn == blockSize) {
							store(C + (i + k) * N + j, acc[k]);
						} else {
							partial_store(C + (i + k) * N + j, acc[k], jn);
						}
					}
				}
			}
		}
	} matrix_t;
} // namespace cortex::_fw::avx2

#endif // CORTEXMIND_CORE_ENGINE_AVX_MATRIX_HPP
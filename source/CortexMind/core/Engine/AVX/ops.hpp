#ifndef CORTEXMIND_CORE_ENGINE_AVX_OPS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_OPS_HPP

#include <CortexMind/core/Engine/AVX/params.hpp>
#include <cmath>

namespace cortex::_fw::avx2 {
	// @brief Load and store operations for vec8f
	// @param item Pointer to the float array to load from
	inline vec8f load(const float* item) {
		return _mm256_load_ps(item);
	}

	// @brief Store operation for vec8f
	// @param dest Pointer to the float array to store to
	// @param item The vec8f item to store
	inline void store(float* dest, vec8f item) {
		_mm256_store_ps(dest, item);
	}

	// @brief Unaligned load and store operations for vec8f
	// @param item Pointer to the float array to load from
	inline vec8f loadu(const float* item) {
		return _mm256_loadu_ps(item);
	}

	// @brief Unaligned store operation for vec8f
	// @param dest Pointer to the float array to store to
	// @param item The vec8f item to store
	inline void storeu(float* dest, vec8f item) {
		_mm256_storeu_ps(dest, item);
	}

	// @brief Partial load operation for vec8f
	// @param item Pointer to the float array to load from
	// @param idx Number of elements to load (0 to 8)
	inline vec8f partial_load(const float* item, const size_t idx) {
		if (idx == 8) return loadu(item);
		int maskVals[8] = {};
		for(size_t i = 0; i < 8; ++i) maskVals[i] = (i < idx) ? -1 : 0;
		const vec8i mask = _mm256_loadu_si256(reinterpret_cast<const vec8i*>(maskVals));
		return _mm256_maskload_ps(item, mask);
	}

	// @brief Partial store operation for vec8f
	// @param dest Pointer to the float array to store to
	// @param item The vec8f item to store
	// @param idx Number of elements to store (0 to 8)
	inline void partial_store(float* dest, vec8f item, const size_t idx) {
		if (idx == 8) {
			storeu(dest, item);
			return;
		}
		int maskVals[8] = {};
		for(size_t i = 0; i < 8; ++i) maskVals[i] = (i < idx) ? -1 : 0;
		const vec8i mask = _mm256_loadu_si256(reinterpret_cast<const vec8i*>(maskVals));
		_mm256_maskstore_ps(dest, mask, item);
	}

	// @brief Arithmetic operations for vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f add(vec8f a, vec8f b) {
		return _mm256_add_ps(a, b);
	}

	// @brief Arithmetic operations for vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f sub(vec8f a, vec8f b) {
		return _mm256_sub_ps(a, b);
	}

	// @brief Arithmetic operations for vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f mul(vec8f a, vec8f b) {
		return _mm256_mul_ps(a, b);
	}

	// @brief Arithmetic operations for vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f div(vec8f a, vec8f b) {
		return _mm256_div_ps(a, b);
	}

	// @brief Broadcast a single float value to all elements of vec8f
	// @param value The float value to broadcast
	inline vec8f broadcast(float value) {
		return _mm256_set1_ps(value);
	}

	// @brief Set individual elements of vec8f
	// @param v0 Element 0
	// @param v1 Element 1
	// @param v2 Element 2
	// @param v3 Element 3
	// @param v4 Element 4
	// @param v5 Element 5
	// @param v6 Element 6
	// @param v7 Element 7
	inline vec8f set(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7) {
		return _mm256_set_ps(v7, v6, v5, v4, v3, v2, v1, v0);
	}

	// @brief Compute the floor of each element in vec8f
	// @param a The input vec8f
	inline vec8f floor(vec8f a) {
		return _mm256_floor_ps(a);
	}

	// @brief Set all elements of vec8f to zero
	inline vec8f zero() {
		return _mm256_setzero_ps();
	}

	// @brief Compute the maximum of two vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f max(vec8f a, vec8f b) {
		return _mm256_max_ps(a, b);
	}

	// @brief Compute the minimum of two vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f min(vec8f a, vec8f b) {
		return _mm256_min_ps(a, b);
	}

	// @brief Compute the square root of each element in vec8f
	// @param a The input vec8f
	inline vec8f sqrt(vec8f a) {
		return _mm256_sqrt_ps(a);
	}

	// @brief Compute the absolute value of each element in vec8f
	// @param a The input vec8f
	inline vec8f abs(vec8f a) {
		const vec8f mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
		return _mm256_and_ps(a, mask);
	}

	// @brief Blend two vec8f based on a mask
	// @param a First operand
	// @param b Second operand
	// @param mask The mask vec8f
	inline vec8f blend(vec8f a, vec8f b, vec8f mask) {
		return _mm256_blendv_ps(a, b, mask);
	}

	// @brief Horizontal add of all elements in vec8f, returning a vec4f with the sums
	// @param a The input vec8f
	inline vec4f horizontal_add(vec8f a) {
		vec8f temp1 = _mm256_hadd_ps(a, a);
		vec8f temp2 = _mm256_hadd_ps(temp1, temp1);
		vec4f low = _mm256_castps256_ps128(temp2);
		vec4f high = _mm256_extractf128_ps(temp2, 1);
		return _mm_add_ps(low, high);
	}

	// @brief Fused multiply-add operation for vec8f (a * b + c)
	// @param a First operand
	// @param b Second operand
	// @param c Third operand
	inline vec8f fmadd(vec8f a, vec8f b, vec8f c) {
		return _mm256_fmadd_ps(a, b, c);
	}

	// @brief Fused multiply-subtract operation for vec8f (a * b - c)
	// @param a First operand
	// @param b Second operand
	// @param c Third operand
	inline vec8f fmsub(vec8f a, vec8f b, vec8f c) {
		return _mm256_fmsub_ps(a, b, c);
	}

	// @brief Blend two vec8f based on a mask
	// @param a First operand
	// @param b Second operand
	// @param mask The mask vec8f
	inline vec8f blendv_ps(vec8f a, vec8f b, vec8f mask) {
		return _mm256_blendv_ps(a, b, mask);
	}

	// @brief Bitwise AND operation for vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f bitwise_and(vec8f a, vec8f b) {
		return _mm256_and_ps(a, b);
	}

	// @brief Bitwise OR operation for vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f bitwise_or(vec8f a, vec8f b) {
		return _mm256_or_ps(a, b);
	}

	// @brief Bitwise XOR operation for vec8f
	// @param a First operand
	// @param b Second operand
	inline vec8f bitwise_xor(vec8f a, vec8f b) {
		return _mm256_xor_ps(a, b);
	}

	// @brief ReLU activation function for vec8f
	// @param a The input vec8f
	inline vec8f relu(vec8f a) {
		return max(a, zero());
	}

	// @brief Approximate exponential function for vec8f using a polynomial approximation
	// @param a The input vec8f
	inline vec8f approx_exg(vec8f a) noexcept {
		vec8f ln2 = broadcast(0.69314718056f);

		vec8f y = div(a, ln2); // 1/ln(2)

		vec8f yi = floor(y);
		vec8f r = sub(a, mul(yi, ln2));

		vec8f c1 = broadcast(1.0f);
		vec8f c2 = broadcast(0.5f);
		vec8f c3 = broadcast(1.0f / 6.0f);
		vec8f c4 = broadcast(1.0f / 24.0f);
		vec8f c5 = broadcast(1.0f / 120.0f);

		vec8f poly = fmadd(c5, r, c4);
		poly = fmadd(poly, r, c3);
		poly = fmadd(poly, r, c2);
		poly = fmadd(poly, r, c1);

		vec8i yi_int = _mm256_cvttps_epi32(yi);      // floor(y) zaten yi
		vec8i exp_bits = _mm256_add_epi32(yi_int, _mm256_set1_epi32(127));
		exp_bits = _mm256_slli_epi32(exp_bits, 23);
		vec8f pow2y = _mm256_castsi256_ps(exp_bits);
		
		return mul(pow2y, poly);
	}

	// @brief Approximate natural logarithm function for vec8f using a polynomial approximation
	// @param a The input vec8f
	inline vec8f approx_log(vec8f a) noexcept {
		const vec8i exponent_mask = _mm256_set1_epi32(0x7F800000);
		const vec8i mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
		const vec8i bias = _mm256_set1_epi32(127);

		vec8i a_int = _mm256_castps_si256(a);
		vec8i exponent = _mm256_srli_epi32(_mm256_and_si256(a_int, exponent_mask), 23);
		vec8i mantissa = _mm256_or_si256(_mm256_and_si256(a_int, mantissa_mask),
							_mm256_set1_epi32(0x3F800000));
		vec8f m = _mm256_castsi256_ps(mantissa);
		vec8f e = _mm256_cvtepi32_ps(_mm256_sub_epi32(exponent, bias));

		const vec8f c1 = broadcast(-0.5f);
		const vec8f c2 = broadcast(1.0f / 3.0f);
		const vec8f c3 = broadcast(-1.0f / 4.0f);
		const vec8f c4 = broadcast(1.0f / 5.0f);
		const vec8f c5 = broadcast(-1.0f / 6.0f);

		vec8f r = sub(m, broadcast(1.0f));
		vec8f r2 = mul(r, r);
		vec8f r3 = mul(r2, r);
		vec8f r4 = mul(r3, r);
		vec8f r5 = mul(r4, r);

		vec8f poly = add(r, mul(r2, c1));
		poly = add(poly, mul(r3, c2));
		poly = add(poly, mul(r4, c3));
		poly = add(poly, mul(r5, c4));

		vec8f ln2 = broadcast(0.69314718056f);
		vec8f result = add(mul(e, ln2), poly);

		const vec8f zero = broadcast(0.0f);
		result = _mm256_blendv_ps(result, broadcast(-INFINITY), _mm256_cmp_ps(a, zero, _CMP_EQ_OQ));
		result = _mm256_blendv_ps(result, broadcast(NAN), _mm256_cmp_ps(a, zero, _CMP_LT_OQ));

		return result;
	}

	// @brief Approximate sigmoid function for vec8f using the approximate exponential function
	// @param x The input vec8f
	inline vec8f approx_sigmoid(vec8f x) noexcept {
		vec8f one = broadcast(1.0f);
		vec8f neg_x = sub(zero(), x);
		vec8f exp_neg_x = approx_exg(neg_x);
		return div(one, add(one, exp_neg_x));
	}

	// @brief Approximate tanh function for vec8f using the approximate exponential function
	// @param x The input vec8f
	inline vec8f approx_tanh(vec8f x) noexcept {
		vec8f two = broadcast(2.0f);
		vec8f neg_two_x = sub(zero(), mul(two, x));
		vec8f exp_neg_two_x = approx_exg(neg_two_x);
		vec8f numerator = sub(broadcast(1.0f), exp_neg_two_x);
		vec8f denominator = add(broadcast(1.0f), exp_neg_two_x);
		return div(numerator, denominator);
	}

	// @brief Compute the sign of each element in vec8f
	// @param a The input vec8f
	inline vec8f sign(vec8f a) {
		const vec8f zero_vec = zero();
		const vec8f one_vec = broadcast(1.0f);
		const vec8f neg_one_vec = broadcast(-1.0f);
		vec8f is_positive = _mm256_cmp_ps(a, zero_vec, _CMP_GT_OQ);
		vec8f is_negative = _mm256_cmp_ps(a, zero_vec, _CMP_LT_OQ);
		vec8f result = blend(zero_vec, one_vec, is_positive);
		result = blend(result, neg_one_vec, is_negative);
		return result;
	}

	// @brief Compute the reciprocal of each element in vec8f
	// @param a The input vec8f
	inline vec8f reciprocal(vec8f a) {
		return _mm256_rcp_ps(a);
	}
} // namespace cortex::_fw::avx2

#endif // CORTEXMIND_CORE_ENGINE_AVX_OPS_HPP
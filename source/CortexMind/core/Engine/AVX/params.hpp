#ifndef CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP
#define CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP

#include <immintrin.h>

namespace cortex::_fw::avx2 {
	using vec4f = __m128;  /// 4 floats in a 128-bit SSE register
	using vec4i = __m128i;  /// 4 int32_t in a 128-bit SSE register
	using vec2d = __m128d;  /// 2 doubles in a 128-bit SSE register
	using vec8f = __m256; /// 8 floats in a 256-bit AVX register
	using vec8i = __m256i; /// 8 int32_t in a 256-bit AVX register
	using vec4d = __m256d; /// 4 doubles in a 256-bit AVX register
} // namespace cortex::_fw::avx2

#endif // CORTEXMIND_CORE_ENGINE_AVX_PARAMS_HPP

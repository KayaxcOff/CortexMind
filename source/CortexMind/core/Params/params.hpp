#ifndef CORTEXMIND_CORE_PARAMS_PARAMS_HPP
#define CORTEXMIND_CORE_PARAMS_PARAMS_HPP

#include <CortexMind/core/Engine/Tensor/tensor.hpp>

namespace cortex {
	using int16 = short; /// Signed 16-bit integer type
	using int32 = int; /// Signed 32-bit integer type
	using int64 = long long; /// Signed 64-bit integer type
	using uint8 = unsigned char; /// Unsigned 8-bit integer type
	using uint16 = unsigned short; /// Unsigned 16-bit integer type
	using uint32 = unsigned int; /// Unsigned 32-bit integer type
	using uint64 = unsigned long long; /// Unsigned 64-bit integer type
	using float32 = float; /// 32-bit floating point type
	using float64 = double; /// 64-bit floating point type
	using byte = unsigned char; /// Byte type
	using char8 = char; /// 8-bit character type
	using tensor = _fw::MindTensor; /// Tensor type alias
} // namespace cortex

#endif // CORTEXMIND_CORE_PARAMS_PARAMS_HPP
//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TYPE_SIZE_HPP
#define CORTEXMIND_FRAMEWORK_TYPE_SIZE_HPP

#include <CortexMind/framework/Type/dtype.hpp>
#include <cwchar>

namespace cortex::_fw {
    [[nodiscard]]
    std::size_t sizeOf(DType type) noexcept;
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TYPE_SIZE_HPP
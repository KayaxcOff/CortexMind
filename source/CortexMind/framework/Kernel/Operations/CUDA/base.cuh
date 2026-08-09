//
// Created by muham on 9.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_KERNEL_OPERATIONS_CUDA_BASE_CUH
#define CORTEXMIND_FRAMEWORK_KERNEL_OPERATIONS_CUDA_BASE_CUH

#include <tlx/concepts.hpp>
#include <tlx/string.hpp>
#include <string_view>

namespace cortex::_fw::ops {
    template<tlx::arithmetic_like T>
    struct kernel_base {
        explicit kernel_base(const tlx::vstring name) {
            this->m_name = name;
        }
        virtual ~kernel_base() = default;

        using type = T;

        [[nodiscard]]
        std::string_view ToString() const noexcept {
            return this->m_name;
        }
    private:
        tlx::vstring m_name;
    };
} //namespace cortex::_fw::ops

#endif //CORTEXMIND_FRAMEWORK_KERNEL_OPERATIONS_CUDA_BASE_CUH
//
// Created by muham on 26.02.2026.
//

#ifndef CORTEXMIND_UTILS_CAST_FACTORY_HPP
#define CORTEXMIND_UTILS_CAST_FACTORY_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex::utils {
    struct TensorFactory {
        TensorFactory();

        void set(float32 _value);
        [[nodiscard]]
        tensor cast(bool _requires_grad = false, _fw::sys::device _dev = _fw::sys::device::host) const;
    private:
        float32 value;
    };
} // namespace cortex::utils

#endif //CORTEXMIND_UTILS_CAST_FACTORY_HPP
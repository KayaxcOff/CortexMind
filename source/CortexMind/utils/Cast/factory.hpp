//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_UTILS_CAST_FACTORY_HPP
#define CORTEXMIND_UTILS_CAST_FACTORY_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/tools/params.hpp>

namespace cortex::utils {
    struct TensorFactory {
        explicit TensorFactory(float32 value = 0.0f);

        [[nodiscard]]
        tensor cast(_fw::sys::deviceType d_tye = _fw::sys::host, boolean requires_grad = false) const;
    private:
        float32 value;
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILS_CAST_FACTORY_HPP
//
// Created by muham on 19.04.2026.
//

#ifndef CORTEXMIND_UTILS_CAST_FACTORY_HPP
#define CORTEXMIND_UTILS_CAST_FACTORY_HPP

#include <CortexMind/framework/Memory/device.hpp>
#include <CortexMind/tools/params.hpp>

namespace cortex::utils {
    /**
     * @brief Factory class for creating scalar tensors with a specific value.
     *
     * Provides a convenient way to create a tensor filled with a constant value,
     * with control over the device (host or CUDA) and whether it requires gradients.
     */
    struct TensorFactory {
        /**
         * @brief Constructs a TensorFactory with the given scalar value.
         * @param value The constant value to fill the resulting tensor with (default: 0.0f)
         */
        explicit TensorFactory(float32 value = 0.0f);

        /**
         * @brief Creates a 1-element tensor filled with the stored value.
         *
         * @param d_type        Target device (host or cuda)
         * @param requires_grad Whether the resulting tensor should track gradients
         * @return A tensor of shape {1} containing the specified value
         */
        [[nodiscard]]
        tensor cast(_fw::sys::deviceType d_type = _fw::sys::host, boolean requires_grad = false) const;
    private:
        float32 value;
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILS_CAST_FACTORY_HPP
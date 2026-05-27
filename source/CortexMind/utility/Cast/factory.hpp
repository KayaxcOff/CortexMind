//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_UTILITY_CAST_FACTORY_HPP
#define CORTEXMIND_UTILITY_CAST_FACTORY_HPP

#include <CortexMind/tools/types.hpp>

namespace cortex::utils {
    /**
     * @brief Factory class for creating scalar tensors with specific properties.
     *
     * Provides a convenient way to create tensors initialized with a single value,
     * with configurable device and gradient tracking settings.
     */
    struct TensorFactory {
        /**
         * @brief Constructs a TensorFactory.
         *
         * @param _dev          Target device for the created tensor (default: HOST)
         * @param requires_grad Whether the created tensor should track gradients
         */
        explicit TensorFactory(_fw::sys::DeviceType _dev = _fw::sys::DeviceType::kHOST, bool requires_grad = false);
        /**
         * @brief Sets the scalar value to be used when creating the tensor.
         *
         * @param value Value that will fill the resulting tensor
         */
        void Set(float32 value);

        /**
         * @brief Creates and returns a tensor filled with the set value.
         *
         * The resulting tensor has shape `{1}` (scalar tensor).
         *
         * @return Tensor of shape `{1}` filled with the configured value
         */
        [[nodiscard]]
        tensor cast() const;
    private:
        float32 m_value;
        _fw::sys::DeviceType m_dev;
        bool m_grad_flag;
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_CAST_FACTORY_HPP
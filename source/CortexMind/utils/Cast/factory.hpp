//
// Created by muham on 19.03.2026.
//

#ifndef CORTEXMIND_UTILS_CAST_FACTORY_HPP
#define CORTEXMIND_UTILS_CAST_FACTORY_HPP

#include <CortexMind/tools/device.hpp>
#include <CortexMind/tools/params.hpp>

namespace cortex::utils {
    /**
     * @brief   Factory class for creating single-value (scalar) tensors
     *
     * Allows setting a float value once and creating multiple tensors on different
     * devices or with different gradient requirements without repeating the value.
     */
    struct TensorFactory {
        /**
         * @brief   Default constructor – initializes value to 0.0f
         */
        TensorFactory();

        /**
         * @brief   Sets the scalar value to be used in subsequent cast() calls
         * @param   _value   New float value
         *
         * @note    Chainable: f.set(1.0f).cast(...)
         */
        void set(float32 _value);
        /**
         * @brief   Creates a new tensor with shape {1} containing the set value
         * @param   d               Target device (host or cuda, default: host)
         * @param   requires_grad   Whether the returned tensor tracks gradients
         * @return  New tensor with single element equal to the stored value
         *
         * @note    Internally calls tensor({1}, d, requires_grad).fill(value)
         * @note    Each call allocates new storage
         */
        [[nodiscard]]
        tensor cast(dev d = dev::host, bool requires_grad = false) const;
    private:
        float32 value; ///< Current scalar value (default: 0.0f)
    };
} // namespace cortex::utils

#endif //CORTEXMIND_UTILS_CAST_FACTORY_HPP
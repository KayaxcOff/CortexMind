//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_DEVICE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_DEVICE_HPP

#include <CortexMind/framework/Memory/type.hpp>
#include <string_view>

namespace cortex::_fw {
    /**
     * @brief Represents the execution device associated with a tensor.
     *
     * TensorDevice is a lightweight wrapper around @ref DeviceType that
     * encapsulates the execution and memory location of tensor data.
     *
     * The class provides utility functions for runtime device inspection
     * and textual representation while allowing future extensions without
     * modifying the tensor interface.
     *
     * Typical use cases include device-aware memory allocation, execution
     * backend selection, debugging, and serialization.
     */
    class TensorDevice {
    public:
        TensorDevice();
        /**
         * @brief Constructs a tensor device from the specified device type.
         *
         * @param type The underlying device type.
         */
        explicit TensorDevice(DeviceType type);
        TensorDevice(const TensorDevice&);
        TensorDevice(TensorDevice&&) noexcept;
        ~TensorDevice();

        /**
         * @brief Returns the underlying device type.
         *
         * @return The stored @ref DeviceType value.
         */
        [[nodiscard]]
        DeviceType type() const noexcept;
        /**
         * @brief Returns the canonical textual representation of the device.
         *
         * @return A string view containing the canonical device name.
         */
        [[nodiscard]]
        std::string_view ToString() const noexcept;

        TensorDevice& operator=(const TensorDevice&);
        TensorDevice& operator=(TensorDevice&&) noexcept;
    private:
        DeviceType m_type;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_DEVICE_HPP
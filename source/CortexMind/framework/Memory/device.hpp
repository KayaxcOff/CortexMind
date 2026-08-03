//
// Created by muham on 3.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_DEVICE_HPP
#define CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_DEVICE_HPP

#include <CortexMind/framework/Memory/type.hpp>
#include <string_view>

namespace cortex::_fw {
    class TensorDevice {
    public:
        explicit TensorDevice(DeviceType type);
        TensorDevice(const TensorDevice&);
        TensorDevice(TensorDevice&&) noexcept;
        ~TensorDevice();

        [[nodiscard]]
        DeviceType type() const noexcept;
        [[nodiscard]]
        std::string_view ToString() const noexcept;

        TensorDevice& operator=(const TensorDevice&);
        TensorDevice& operator=(TensorDevice&&) noexcept;
    private:
        DeviceType m_type;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_MEMORY_DEVICE_DEVICE_HPP
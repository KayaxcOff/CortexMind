//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_UTILITY_DATA_FRAME_COLUMN_HPP
#define CORTEXMIND_UTILITY_DATA_FRAME_COLUMN_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/tools/values.hpp>
#include <CortexMind/tools/types.hpp>

namespace cortex::utils {
    class Column {
    public:
        Column(std::vector<float32>& data, std::string  name);
        ~Column();

        [[nodiscard]]
        size_t size() const;
        [[nodiscard]]
        const std::string& name() const;
        [[nodiscard]]
        tensor toTensor(_fw::sys::DeviceType dev = host) const;

        float32& operator[](size_t i);
        const float32 &operator[](size_t i) const;
    private:
        std::vector<float32>& m_data;
        std::string m_name;
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_DATA_FRAME_COLUMN_HPP
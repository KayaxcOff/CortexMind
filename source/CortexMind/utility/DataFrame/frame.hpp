//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_UTILITY_DATA_FRAME_FRAME_HPP
#define CORTEXMIND_UTILITY_DATA_FRAME_FRAME_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/utility/DataFrame/column.hpp>
#include <unordered_map>
#include <tuple>
#include <vector>

namespace cortex::utils {
    class DataFrame {
    public:
        explicit DataFrame(const std::string& path);
        DataFrame();
        ~DataFrame();

        Column operator[](const std::string& col);
        Column operator[](const std::string& col) const;

        size_t rows() const;
        size_t cols() const;
        bool hasCol(const std::string& col) const;
        const std::vector<std::string>& columnNames() const;

        void head(size_t n = 5) const;;
        void info() const;
        bool isNan() const;

        void normalize(const std::string& col);
        void normalize();
        void scale(float32 value);
        void dropNan();

        void oneHot(const std::string& col);

        [[nodiscard]]
        std::pair<tensor, tensor> toTensor(const std::string& target, _fw::sys::DeviceType dev = host) const;
        [[nodiscard]]
        std::tuple<tensor, tensor, tensor, tensor> split(const std::string& target, float32 ratio = 0.8f, _fw::sys::DeviceType dev = host) const;
    private:
        std::vector<std::string> m_columns;
        std::unordered_map<std::string, std::vector<float32>> m_data;
        size_t m_rows = 0;

        void parseCsv(const std::string& path);
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_DATA_FRAME_FRAME_HPP
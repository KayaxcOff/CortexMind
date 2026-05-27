//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_UTILITY_DATA_FRAME_FRAME_HPP
#define CORTEXMIND_UTILITY_DATA_FRAME_FRAME_HPP

#include <CortexMind/framework/Tools/series.hpp>
#include <CortexMind/tools/types.hpp>
#include <unordered_map>
#include <string>
#include <vector>
#include <utility>

namespace cortex::utils {
    class DataFrame {
    public:
        explicit DataFrame(const std::string& path);
        ~DataFrame();

        void Set(const std::string &target);
        void drop(const std::string& name);

        [[nodiscard]]
        bool is_nan();
        [[nodiscard]]
        int64 row() const;
        [[nodiscard]]
        int64 col() const;
        [[nodiscard]]
        std::pair<tensor, tensor> split(float32 ratio);
        [[nodiscard]]
        _fw::Series operator[](const std::string& name);
        [[nodiscard]]
        std::pair<tensor, tensor> toTensor();
    private:
        tensor t;
        int64 m_col, m_row;
        std::string target_name;
        std::vector<std::string> m_order;
        std::unordered_map<std::string, std::vector<float32>> idx;

        void load_csv(const std::string& path);
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_DATA_FRAME_FRAME_HPP
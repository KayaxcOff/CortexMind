//
// Created by muham on 4.03.2026.
//

#ifndef CORTEXMIND_DATASET_FRAME_DF_HPP
#define CORTEXMIND_DATASET_FRAME_DF_HPP

#include <CortexMind/tools/params.hpp>
#include <utility>
#include <vector>

namespace cortex::ds {
    class DataFrame {
    public:
        DataFrame();
        ~DataFrame();
        
        void info() const;

        [[nodiscard]]
        int64 rows() const;
        [[nodiscard]]
        int64 cols() const;
        [[nodiscard]]
        int64 nan_values() const;
        [[nodiscard]]
        tensor drop(const string& col) const;
        [[nodiscard]]
        tensor drop(int64 col_idx) const;
        [[nodiscard]]
        std::vector<string> columns() const;
        [[nodiscard]]
        std::pair<tensor, tensor> split(const string& target) const;
        [[nodiscard]]
        std::pair<tensor, tensor> split(int64 target_idx) const;
        [[nodiscard]]
        DataFrame dropColumn(const string& col) const;
        [[nodiscard]]
        DataFrame dropColumn(int64 col_idx) const;

        static DataFrame from_csv(const string& path, bool header = true);
        static DataFrame from_json(const string& path);

        tensor operator[](const string& col) const;
        tensor operator[](int64 col_idx) const;
    private:
        tensor t;
        std::vector<string> column;
    };
} // namespace cortex::ds

#endif //CORTEXMIND_DATASET_FRAME_DF_HPP
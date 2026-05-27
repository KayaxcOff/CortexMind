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
    /**
     * @brief Simple DataFrame class for handling tabular data (CSV-like).
     *
     * Provides basic functionality for loading data from CSV files, column access,
     * statistical information, and splitting into features (X) and target (Y).
     */
    class DataFrame {
    public:
        /**
         * @brief Constructs a DataFrame by loading data from a CSV file.
         *
         * @param path Path to the CSV file
         */
        explicit DataFrame(const std::string& path);
        ~DataFrame();

        /**
         * @brief Sets the target (label) column for supervised learning.
         *
         * @param idx Name of the target column
         */
        void Set(const std::string &idx);
        /**
         * @brief Drops a column from the DataFrame.
         *
         * @param idx Name of the column to drop
         */
        void drop(const std::string& idx);
        /**
         * @brief Prints basic information about the DataFrame.
         */
        void info() const;
        /**
         * @brief Displays the first N rows of the DataFrame.
         *
         * @param row_to_show Number of rows to display (default: 5)
         */
        void head(size_t row_to_show = 5);
        /**
         * @brief Checks if the DataFrame contains any NaN values.
         *
         * @return `true` if any NaN value is found, `false` otherwise
         */
        [[nodiscard]]
        bool NaN();
        /**
         * @brief Returns the number of rows.
         */
        [[nodiscard]]
        int64 row() const;
        /**
         * @brief Returns the number of columns.
         */
        [[nodiscard]]
        int64 col() const;
        /**
         * @brief Splits the DataFrame into features (X) and target (Y).
         *
         * Requires `Set()` to be called beforehand to specify the target column.
         *
         * @return Pair of tensors: (X features, Y target)
         */
        [[nodiscard]]
        std::pair<tensor, tensor> split();
        /**
         * @brief Access a column by name (mutable).
         */
        _fw::Series& operator[](const std::string& idx);
        /**
         * @brief Access a column by name (const).
         */
        const _fw::Series& operator[](const std::string& idx) const;
    private:
        std::unordered_map<std::string, _fw::Series> series;
        std::vector<std::string> names;
        int64 m_col, m_row;
        std::string target;
        bool isInit;
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_DATA_FRAME_FRAME_HPP
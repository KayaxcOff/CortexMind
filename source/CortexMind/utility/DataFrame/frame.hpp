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
     * @brief DataFrame for loading, exploring, and preprocessing tabular data.
     *
     * Supports CSV loading, multiple target columns, one-hot encoding,
     * label encoding, boolean encoding, and splitting into features (X) and targets (Y).
     */
    class DataFrame {
    public:
        /**
         * @brief Constructs a DataFrame by loading data from a CSV file.
         *
         * The first row is treated as the header (column names).
         *
         * @param path Path to the CSV file
         */
        explicit DataFrame(const std::string& path);
        ~DataFrame();

        /**
         * @brief Sets a single target column for supervised learning.
         *
         * @param idx Name of the target column
         */
        void Set(const std::string &idx);
        /**
         * @brief Sets multiple target columns for multi-output supervised learning.
         *
         * @param idx List of target column names
         */
        void Set(const std::vector<std::string>& idx);
        /**
         * @brief Drops a column from the DataFrame.
         *
         * @param idx Name of the column to drop
         */
        void drop(const std::string& idx);
        /**
         * @brief Prints basic information about the DataFrame (shape and column types).
         */
        void info() const;
        /**
         * @brief Displays the first N rows of the DataFrame.
         *
         * @param row_to_show Number of rows to display (default: 5)
         */
        void head(size_t row_to_show = 5) const;
        /**
         * @brief Performs one-hot encoding on a categorical (string) column.
         *
         * Creates new binary columns for each unique category.
         *
         * @param idx Name of the column to encode
         */
        void one_hot(const std::string& idx);
        /**
         * @brief Converts a boolean column to float32 (0.0 / 1.0).
         *
         * @param idx Name of the boolean column
         */
        void encode_bool(const std::string& idx);
        /**
         * @brief Performs label encoding on a string column.
         *
         * Maps each unique string to an integer (0, 1, 2, ...).
         *
         * @param idx Name of the string column
         */
        void label_encode(const std::string& idx);
        void shuffle();

        /**
         * @brief Checks if the DataFrame contains any NaN values.
         *
         * @return `true` if any NaN is found, `false` otherwise
         */
        [[nodiscard]]
        bool NaN() const;
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
         * @brief Splits the DataFrame into features (X) and target(s) (Y).
         *
         * Requires `Set()` to be called first to specify target column(s).
         *
         * @return Pair of tensors: `(X features, Y target(s))`
         */
        [[nodiscard]]
        std::pair<tensor, tensor> split();
        /**
         * @brief Splits the DataFrame into features (X_Train, X_Test) and target(s) (Y_Train, Y_Test).
         *
         * Requires `Set()` to be called first to specify target column(s).
         *
         * @return Pair of tensors: `(X features, Y target(s))`
         */
        [[nodiscard]]
        std::pair<std::pair<tensor, tensor>, std::pair<tensor, tensor>> train_test_split(float32 test_size, bool shuffle_data);

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
        std::vector<std::string> targets;
        bool isInit;
        std::vector<size_t> m_indices;
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_DATA_FRAME_FRAME_HPP
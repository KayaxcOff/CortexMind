//
// Created by muham on 4.03.2026.
//

#include "CortexMind/dataset/Frame/df.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace cortex::ds;
using namespace cortex;

DataFrame::DataFrame() = default;

DataFrame::~DataFrame() = default;

void DataFrame::info() const {
    std::cout << "[CXM - DataFrame]" << std::endl;
    std::cout << "Shape: (" << this->rows() << ", " << this->cols() << ")" << std::endl;
    std::cout << "Columns: ";
    for (size_t i = 0; i < this->column.size(); ++i) {
        std::cout << this->column[i];
        if (i + 1 < this->column.size()) std::cout << ", ";
    }
    std::cout << "\n";
}

int64 DataFrame::rows() const {
    return this->t.shape()[0];
}

int64 DataFrame::cols() const {
    return this->t.shape()[1];
}

int64 DataFrame::nan_values() const {
    int64 output = 0;
    for (int64 i = 0; i < this->rows(); ++i) {
        for (int64 j = 0; j < this->cols(); ++j) {
            if (std::isnan(this->t.at(i, j))) output++;
        }
    }
    return output;
}

tensor DataFrame::drop(const string &col) const {
    const auto it = std::ranges::find(this->column, col);
    CXM_ASSERT(it != this->column.end(), "cortex::ds::DataFrame::drop()", "Column not found.");
    return drop(std::distance(this->column.begin(), it));
}

tensor DataFrame::drop(const int64 col_idx) const {
    CXM_ASSERT(col_idx >= 0 && col_idx < this->cols(), "cortex::ds::DataFrame::drop()", "Column index out of range.");

    const int64 out_cols = this->cols() - 1;
    tensor output({this->rows(), out_cols}, false);

    for (int64 i = 0; i < this->rows(); ++i) {
        int64 out_j = 0;
        for (int64 j = 0; j < this->cols(); ++j) {
            if (j == col_idx) continue;
            output.at(i, out_j++) = this->t.at(i, j);
        }
    }
    return output;
}

std::vector<string> DataFrame::columns() const {
    return this->column;
}

std::pair<tensor, tensor> DataFrame::split(const string &target) const {
    const auto it = std::ranges::find(this->column, target);
    CXM_ASSERT(it != this->column.end(),
               "cortex::ds::DataFrame::split()", "Column not found.");
    return split(std::distance(this->column.begin(), it));
}

std::pair<tensor, tensor> DataFrame::split(const int64 target_idx) const {
    CXM_ASSERT(target_idx >= 0 && target_idx < this->cols(), "cortex::ds::DataFrame::split()", "Column index out of range.");
    return {this->drop(target_idx), (*this)[target_idx]};
}

DataFrame DataFrame::dropColumn(const string &col) const {
    const auto it = std::ranges::find(this->column, col);
    CXM_ASSERT(it != this->column.end(), "cortex::ds::DataFrame::dropColumn()", "Column not found.");
    return dropColumn(std::distance(this->column.begin(), it));
}

DataFrame DataFrame::dropColumn(const int64 col_idx) const {
    CXM_ASSERT(col_idx >= 0 && col_idx < this->cols(), "cortex::ds::DataFrame::dropColumn()", "Column index out of range.");

    DataFrame df;
    df.column = this->column;
    df.column.erase(df.column.begin() + col_idx);

    df.t = tensor({this->rows(), static_cast<int64>(df.column.size())}, false);

    for (int64 i = 0; i < this->rows(); ++i) {
        int64 out_j = 0;
        for (int64 j = 0; j < this->cols(); ++j) {
            if (j == col_idx) continue;
            df.t.at(i, out_j++) = this->t.at(i, j);
        }
    }
    return df;
}

DataFrame DataFrame::from_csv(const string &path, bool header) {
    std::ifstream ifs(path);
    CXM_ASSERT(ifs.is_open(), "cortex::ds::DataFrame::from_csv()", "File cannot be opened.");

    DataFrame df;
    std::vector<float32> values;
    string line;
    size_t n_cols = 0;
    size_t n_rows = 0;

    if (header && std::getline(ifs, line)) {
        std::stringstream ss(line);
        string cell;
        while (std::getline(ss, cell, ',')) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            df.column.push_back(cell);
        }
        n_cols = df.column.size();
    }

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        string cell;
        size_t current_cols = 0;

        while (std::getline(ss, cell, ',')) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            values.push_back(std::stof(cell));
            current_cols++;
        }

        if (n_cols == 0) n_cols = current_cols;
        else CXM_ASSERT(n_cols == current_cols, "cortex::ds::DataFrame::from_csv()", "CSV is not rectangular.");
        n_rows++;
    }

    CXM_ASSERT(n_rows > 0 && n_cols > 0, "cortex::ds::DataFrame::from_csv()", "CSV is empty.");

    if (!header) {
        df.column.clear();
        for (size_t i = 0; i < n_cols; ++i)
            df.column.push_back("col_" + std::to_string(i));
    }

    df.t = tensor({static_cast<int64>(n_rows), static_cast<int64>(n_cols)}, values.data(), false);
    return df;
}

DataFrame DataFrame::from_json(const string &path) {
    std::ifstream ifs(path);
    CXM_ASSERT(ifs.is_open(), "cortex::ds::DataFrame::from_json()", "File cannot be opened.");

    nlohmann::json j;
    ifs >> j;

    CXM_ASSERT(j.contains("columns"), "cortex::ds::DataFrame::from_json()", "Missing 'columns' key.");
    CXM_ASSERT(j.contains("data"),    "cortex::ds::DataFrame::from_json()", "Missing 'data' key.");

    DataFrame df;
    df.column = j["columns"].get<std::vector<string>>();

    const std::vector<std::vector<float32>> rows = j["data"].get<std::vector<std::vector<float32>>>();
    const size_t n_rows = rows.size();
    const size_t n_cols = df.column.size();

    std::vector<float32> values;
    values.reserve(n_rows * n_cols);
    for (const auto& row : rows) {
        CXM_ASSERT(row.size() == n_cols,
                   "cortex::ds::DataFrame::from_json()", "Row size mismatch.");
        for (const auto v : row) values.push_back(v);
    }

    df.t = tensor({static_cast<int64>(n_rows), static_cast<int64>(n_cols)},
                  values.data(), false);
    return df;
}

tensor DataFrame::operator[](const string &col) const {
    const auto it = std::ranges::find(this->column, col);
    CXM_ASSERT(it != this->column.end(),
               "cortex::ds::DataFrame::operator[]()", "Column not found.");
    return (*this)[std::distance(this->column.begin(), it)];
}

tensor DataFrame::operator[](const int64 col_idx) const {
    CXM_ASSERT(col_idx >= 0 && col_idx < this->cols(), "cortex::ds::DataFrame::operator[]()", "Column index out of range.");

    tensor output({this->rows(), 1}, false);
    for (int64 i = 0; i < this->rows(); ++i)
        output.at(i, 0) = this->t.at(i, col_idx);
    return output;
}
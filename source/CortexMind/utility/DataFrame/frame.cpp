//
// Created by muham on 27.05.2026.
//

#include "CortexMind/utility/DataFrame/frame.hpp"
#include <CortexMind/framework/Tools/as_string.hpp>
#include <CortexMind/tools/values.hpp>
#include <fstream>
#include <iostream>
#include <ranges>
#include <sstream>

using namespace cortex::_fw;
using namespace cortex::utils;
using namespace cortex;

DataFrame::DataFrame(const std::string& path) : m_col(0), m_row(0), isInit(false) {
    std::ifstream file(path);
    CXM_ASSERT(!file.is_open(), "Can't open " + path + " file");

    std::string line;

    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            this->names.push_back(cell);
            this->series[cell] = Series();
        }
    }

    this->m_col = static_cast<int64>(this->names.size());

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        size_t col_idx = 0;

        while (std::getline(ss, cell, ',')) {
            if (col_idx < this->names.size()) {
                const std::string& col_name = this->names[col_idx];

                try {
                    f32 val = std::stof(cell);
                    if (this->series[col_name].dtype() == DType::Float32) {
                        this->series[col_name].as<f32>().push_back(val);
                    }
                } catch (const std::invalid_argument&) {
                    if (cell == "true" || cell == "false" || cell == "True" || cell == "False") {
                        if (this->series[col_name].dtype() == DType::Float32 && this->series[col_name].empty()) {
                            this->series[col_name] = Series(std::vector<bool>{});
                        }
                        this->series[col_name].as<bool>().push_back(cell == "true" || cell == "True");
                    } else {
                        if (this->series[col_name].dtype() == DType::Float32 && this->series[col_name].empty()) {
                            this->series[col_name] = Series(std::vector<std::string>{});
                        }
                        this->series[col_name].as<std::string>().push_back(cell);
                    }
                }
            }
            col_idx++;
        }
        this->m_row++;
    }

    file.close();
}

DataFrame::~DataFrame() = default;

void DataFrame::Set(const std::string &idx) {
    this->targets = {idx};
    this->isInit = true;
}

void DataFrame::Set(const std::vector<std::string> &idx) {
    this->targets = idx;
    this->isInit = true;
}

void DataFrame::drop(const std::string& idx) {
    this->series.erase(idx);
    std::erase(this->names, idx);
    --this->m_col;
}

void DataFrame::info() const {
    std::cout << "<DataFrame: " << this->m_row << " rows x " << this->m_col << " cols>\n";
    for (const auto& item : this->names) {
        const auto& it = this->series.at(item);
        std::string type_str;
        switch (it.dtype()) {
            case DType::Float32: type_str = as_string(DType::Float32); break;
            case DType::Bool:    type_str = as_string(DType::Bool);    break;
            case DType::String:  type_str = as_string(DType::String);  break;
        }
        std::cout << " - " << item << " (" << type_str << ")\n";
    }
}

void DataFrame::head(const size_t row_to_show) const {
    for (const auto& item : this->names) {
        std::cout << item << "\t";
    }
    std::cout << "\n";

    const size_t limit = std::min(row_to_show, static_cast<size_t>(this->m_row));
    for (size_t i = 0; i < limit; ++i) {
        for (const auto& item : this->names) {
            switch (const auto& it = this->series.at(item); it.dtype()) {
                case DType::Float32:
                    std::cout << it.as<f32>()[i];
                    break;
                case DType::Bool:
                    std::cout << it.as<bool>()[i];
                    break;
                case DType::String:
                    std::cout << it.as<std::string>()[i];
                    break;
            }
            std::cout << "\t";
        }
        std::cout << "\n";
    }
}

bool DataFrame::NaN() const {
    for (const auto& item : this->names) {
        const auto& s = this->series.at(item);
        if (s.dtype() != DType::Float32) {
            continue;
        }
        for (const f32 it : s.as<f32>()) {
            if (std::isnan(it)) {
                return true;
            }
        }
    }
    return false;
}

int64 DataFrame::row() const {
    return this->m_row;
}

int64 DataFrame::col() const {
    return this->m_col;
}
/*
std::pair<tensor, tensor> DataFrame::split() {
    CXM_ASSERT(!this->isInit, "Target column is not initialized. Call Set() first.");

    for (const auto& item : this->targets) {
        CXM_ASSERT(!this->series.contains(item), "Target column not found in DataFrame.");
    }

    auto y_cols = static_cast<int64>(this->targets.size());
    int64 x_cols = this->m_col - y_cols;

    tensor _x({this->m_row, x_cols}, host);
    tensor _y({this->m_row, y_cols}, host);

    std::vector<float32> x_data;
    std::vector<float32> y_data;

    x_data.reserve(x_cols * this->m_row);
    y_data.reserve(y_cols * this->m_row);

    for (size_t i = 0; i < this->m_row; ++i) {
        for (const auto& item : this->names) {
            if (std::ranges::find(this->targets, item) == this->targets.end()) {
                x_data.push_back(this->series[item].data()[i]);
            } else {
                y_data.push_back(this->series[item].data()[i]);
            }
        }
    }

    _x.SetData(x_data.data());
    _y.SetData(y_data.data());

    return std::make_pair(_x, _y);
}
*/

std::pair<tensor, tensor> DataFrame::split() {
    CXM_ASSERT(!this->isInit, "Target column is not initialized. Call Set() first.");

    for (const auto& item : this->targets) {
        CXM_ASSERT(!this->series.contains(item), "Target column not found in DataFrame.");
    }

    auto y_cols = static_cast<int64>(this->targets.size());
    int64 x_cols = this->m_col - y_cols;

    tensor _x({this->m_row, x_cols}, host);
    tensor _y({this->m_row, y_cols}, host);

    std::vector<float32> x_data;
    std::vector<float32> y_data;

    x_data.reserve(x_cols * this->m_row);
    y_data.reserve(y_cols * this->m_row);

    for (size_t i = 0; i < this->m_row; ++i) {
        for (const auto& item : this->names) {
            if (std::ranges::find(this->targets, item) == this->targets.end()) {
                x_data.push_back(this->series[item].data()[i]);
            }
        }

        for (const auto& item : this->targets) {
            y_data.push_back(this->series[item].data()[i]);
        }
    }

    _x.SetData(x_data.data());
    _y.SetData(y_data.data());

    return std::make_pair(_x, _y);
}

void DataFrame::one_hot(const std::string& idx) {
    if (this->series[idx].dtype() == DType::Float32) {
        std::vector<std::string> str_vec;
        const auto& float_data = this->series[idx].as<f32>();
        str_vec.reserve(float_data.size());

        for (const auto item : float_data) {
            str_vec.push_back(std::to_string(item));
        }
        this->series[idx] = Series(std::move(str_vec));
    }

    CXM_ASSERT(this->series[idx].dtype() != DType::String, "one_hot() is only valid for string sequences: " + idx);

    const auto& str_vec = this->series[idx].as<std::string>();

    std::vector<std::string> categories;
    for (const auto& val : str_vec) {
        if (std::ranges::find(categories, val) == categories.end()) {
            categories.push_back(val);
        }
    }

    std::vector<std::string> generated_cols;


    for (const auto& item : categories) {
        std::string new_col;
        new_col.reserve(idx.size() + 1 + item.size());
        new_col += idx;
        new_col += '_';
        new_col += item;

        generated_cols.push_back(new_col);

        std::vector<f32> encoded(str_vec.size());
        for (size_t i = 0; i < str_vec.size(); ++i) {
            encoded[i] = (str_vec[i] == item) ? 1.0f : 0.0f;
        }

        this->series[new_col] = Series(std::move(encoded));
        this->names.push_back(new_col);
        ++this->m_col;
    }

    this->drop(idx);
}

void DataFrame::encode_bool(const std::string& idx) {
    CXM_ASSERT(this->series[idx].dtype() != DType::Bool, "encode_bool() is only valid for Boolean series: " + idx);

    const auto& bool_vec = this->series[idx].as<bool>();
    std::vector<f32> encoded(bool_vec.size());
    for (size_t i = 0; i < bool_vec.size(); ++i) {
        encoded[i] = bool_vec[i] ? 1.0f : 0.0f;
    }

    this->series[idx] = Series(std::move(encoded));
}

void DataFrame::label_encode(const std::string& idx) {
    CXM_ASSERT(this->series[idx].dtype() != DType::String, "label_encode() yalnızca String seriler için geçerli: " + idx);

    const auto& str_vec = this->series[idx].as<std::string>();

    std::vector<std::string> categories;
    for (const auto& item : str_vec) {
        if (std::ranges::find(categories, item) == categories.end()) {
            categories.push_back(item);
        }
    }

    std::vector<f32> encoded(str_vec.size());
    for (size_t i = 0; i < str_vec.size(); ++i) {
        const auto it = std::ranges::find(categories, str_vec[i]);
        encoded[i] = static_cast<f32>(std::distance(categories.begin(), it));
    }

    this->series[idx] = Series(std::move(encoded));
}
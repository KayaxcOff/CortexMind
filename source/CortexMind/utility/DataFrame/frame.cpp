//
// Created by muham on 27.05.2026.
//

#include "CortexMind/utility/DataFrame/frame.hpp"
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
                try {
                    float val = std::stof(cell);
                    this->series[this->names[col_idx]].data().push_back(val);
                } catch (const std::invalid_argument&) {
                    this->series[this->names[col_idx]].data().push_back(0.0f);
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
    this->target = idx;
    this->isInit = true;
}

void DataFrame::drop(const std::string& idx) {
    this->series.erase(idx);
    std::erase(this->names, idx);
    --this->m_col;
}

void DataFrame::info() const {
    std::cout << "<DataFrame: " << this->m_row << " rows x " << this->m_col << " cols>\n";
    for (const auto& name : this->names) {
        std::cout << " - " << name << " (float32)\n";
    }
}

void DataFrame::head(const size_t row_to_show) {
    for (const auto& item : this->names) {
        std::cout << item << "\t";
    }
    std::cout << "\n";

    const size_t limit = std::min(row_to_show, static_cast<size_t>(this->m_row));
    for (size_t i = 0; i < limit; ++i) {
        for (const auto& item : this->names) {
            std::cout << this->series[item].data()[i] << "\t";
        }
        std::cout << "\n";
    }
}

bool DataFrame::NaN() {
    for (const auto &vals: this->series | std::views::values) {
        for (const float32 v : vals.data()) {
            if (std::isnan(v)) {
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

std::pair<tensor, tensor> DataFrame::split() {
    CXM_ASSERT(!isInit, "Target column is not initialized. Call Set() first.");
    CXM_ASSERT(!this->series.contains(this->target), "Target column not found in DataFrame.");

    int64 x_cols = this->m_col - 1;

    tensor _x({x_cols, this->m_row}, host);
    tensor _y({1, this->m_row}, host);

    _y.SetData(this->series[this->target].data().data());

    std::vector<float> x_data;
    x_data.reserve(x_cols * this->m_row);

    for (const auto& name : this->names) {
        if (name == this->target) {
            continue;
        }

        const auto& col_vector = this->series[name].data();
        x_data.insert(x_data.end(), col_vector.begin(), col_vector.end());
    }

    _x.SetData(x_data.data());

    return std::make_pair(_x, _y);
}
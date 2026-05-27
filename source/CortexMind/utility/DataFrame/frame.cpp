//
// Created by muham on 27.05.2026.
//

#include "CortexMind/utility/DataFrame/frame.hpp"
#include <CortexMind/tools/values.hpp>
#include <fstream>
#include <ranges>
#include <sstream>

using namespace cortex::_fw;
using namespace cortex::utils;
using namespace cortex;

DataFrame::DataFrame(const std::string& path) : m_col(0), m_row(0) {
    this->load_csv(path);
}

DataFrame::~DataFrame() = default;

void DataFrame::Set(const std::string &target) {
    CXM_ASSERT(!this->idx.contains(target), "Column not found: " + target);
    this->target_name = target;
}

void DataFrame::drop(const std::string& name) {
    this->idx.erase(name);
    std::erase(this->m_order, name);
    --this->m_col;
}

bool DataFrame::is_nan() {
    for (const auto &vals: this->idx | std::views::values) {
        for (const float32 v : vals) {
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

void DataFrame::load_csv(const std::string &path) {
    std::ifstream file(path);
    CXM_ASSERT(!file.is_open(), "Cannot open file: " + path);

    std::string line;

    std::getline(file, line);
    std::stringstream ss(line);
    std::string col;
    while (std::getline(ss, col, ',')) {
        std::erase(col, '\r');
        this->m_order.push_back(col);
        this->idx[col] = {};
    }
    this->m_col = static_cast<int64>(this->m_order.size());

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream row_ss(line);
        std::string val;
        size_t c = 0;
        while (std::getline(row_ss, val, ',')) {
            std::erase(val, '\r');
            if (c < this->m_order.size()) {
                try {
                    this->idx[this->m_order[c]].push_back(std::stof(val));
                } catch (...) {
                    this->idx[this->m_order[c]].push_back(std::numeric_limits<float32>::quiet_NaN());
                }
            }
            ++c;
        }
        ++this->m_row;
    }
}

Series DataFrame::operator[](const std::string& name) {
    CXM_ASSERT(!this->idx.contains(name),
        "Column not found: " + name);
    return Series(this->idx[name]);
}

std::pair<tensor, tensor> DataFrame::toTensor() {
    CXM_ASSERT(this->target_name.empty(), "Target not set, call Set() first");

    std::vector<float32> X_data, Y_data;
    const auto N = static_cast<size_t>(this->m_row);

    for (size_t r = 0; r < N; ++r) {
        for (const auto& col : this->m_order) {
            if (col == this->target_name) continue;
            X_data.push_back(this->idx[col][r]);
        }
        Y_data.push_back(this->idx[this->target_name][r]);
    }

    const int64 n_features = this->m_col - 1;
    tensor X({static_cast<int64>(N), n_features}, X_data.data(), host);
    tensor Y({static_cast<int64>(N), 1},Y_data.data(), host);

    return {X, Y};
}

std::pair<tensor, tensor> DataFrame::split(const float32 raite) {
    const int64 split_idx = this->m_row * static_cast<int64>(raite);

    auto [X, Y] = this->toTensor();

    tensor X_train = X.slice(0, 0, split_idx);
    tensor X_test  = X.slice(0, split_idx, this->m_row);
    tensor Y_train = Y.slice(0, 0, split_idx);
    tensor Y_test  = Y.slice(0, split_idx, this->m_row);

    return {X_train.clone(), X_test.clone()};
}
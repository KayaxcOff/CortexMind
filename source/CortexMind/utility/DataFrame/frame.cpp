//
// Created by muham on 27.05.2026.
//

#include "CortexMind/utility/DataFrame/frame.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <ranges>
#include <xutility>

using namespace cortex::utils;
using namespace cortex::_fw::sys;
using namespace cortex;

DataFrame::DataFrame(const std::string &path) {
    this->parseCsv(path);
}

DataFrame::DataFrame() = default;

DataFrame::~DataFrame() = default;

size_t DataFrame::rows() const {
    return this->m_rows;
}

size_t DataFrame::cols() const {
    return this->m_columns.size();
}

bool DataFrame::hasCol(const std::string &col) const {
    return this->m_data.contains(col);
}

const std::vector<std::string>& DataFrame::columnNames() const {
    return this->m_columns;
}

void DataFrame::head(size_t n) const {
    n = std::min(n, this->m_rows);

    for (const auto& item : this->m_columns) {
        std::cout << item << "\t";
    }

    std::cout << "\n";

    for (size_t i = 0; i < n; i++) {
        for (const auto& item : this->m_columns) {
            std::cout << this->m_data.at(item)[i] << "\t";
        }
        std::cout << "\n";
    }
}

void DataFrame::info() const {
    std::cout << "Rows: " << rows() << "\n";
    std::cout << "Cols: " << cols() << "\n";

    for (const auto& item : m_columns) {
        std::cout << item
                  << " -> "
                  << m_data.at(item).size()
                  << "\n";
    }
}

bool DataFrame::isNan() const {
    for (const auto &item: this->m_data | std::views::values) {
        for (const float32 it : item) {
            if (std::isnan(it)) {
                return true;
            }
        }
    }
    return false;
}

void DataFrame::normalize(const std::string &col) {
    auto& vec = m_data[col];

    const float32 minVal = *std::ranges::min_element(vec);
    const float32 maxVal = *std::ranges::max_element(vec);

    const float32 range = maxVal - minVal;

    if (range == 0.0f) {
        return;
    }

    for (auto& v : vec) {
        v = (v - minVal) / range;
    }
}

void DataFrame::normalize() {
    for (const auto& item : this->m_columns) {
        this->normalize(item);
    }
}

void DataFrame::scale(const float32 value) {
    for (auto &vec: this->m_data | std::views::values) {
        for (auto& v : vec) {
            v *= value;
        }
    }
}

void DataFrame::dropNan() {
    std::vector<size_t> validRows;

    for (size_t i = 0; i < this->m_rows; i++) {
        bool valid = true;

        for (const auto& col : this->m_columns) {
            if (std::isnan(this->m_data[col][i])) {
                valid = false;
                break;
            }
        }

        if (valid) {
            validRows.push_back(i);
        }
    }

    for (auto &vec: this->m_data | std::views::values) {
        std::vector<float32> cleaned;

        for (const size_t idx : validRows) {
            cleaned.push_back(vec[idx]);
        }

        vec = std::move(cleaned);
    }

    this->m_rows = validRows.size();
}

void DataFrame::oneHot(const std::string &col) {

}

std::pair<tensor, tensor> DataFrame::toTensor(const std::string& target, const DeviceType dev) const {

    std::vector<float32> xData;
    std::vector<float32> yData;

    const size_t featureCount = this->cols() - 1;

    for (size_t i = 0; i < this->m_rows; i++) {

        for (const auto& item : this->m_columns) {
            if (item == target) {
                continue;
            }

            xData.push_back(this->m_data.at(item)[i]);
        }

        yData.push_back(this->m_data.at(target)[i]);
    }

    tensor X(
        {static_cast<int64>(this->m_rows), static_cast<int64>(featureCount)},
        xData.data(),
        dev
    );

    tensor Y(
        {static_cast<int64>(this->m_rows), 1},
        yData.data(),
        dev
    );

    return {X, Y};
}

std::tuple<tensor, tensor, tensor, tensor> DataFrame::split(const std::string &target, float32 ratio, DeviceType dev) const {
    auto [X, Y] = toTensor(target, dev);

    size_t trainSize = this->m_rows * ratio;
    size_t testSize  = this->m_rows - trainSize;

    size_t featureCount = cols() - 1;

    std::vector<float32> xTrain;
    std::vector<float32> yTrain;
    std::vector<float32> xTest;
    std::vector<float32> yTest;

    for (size_t i = 0; i < this->m_rows; i++) {

        bool train = i < trainSize;

        for (size_t j = 0; j < featureCount; j++) {

            float32 val = X.at(i * featureCount + j);

            if (train) {
                xTrain.push_back(val);
            } else {
                xTest.push_back(val);
            }
        }

        if (train) {
            yTrain.push_back(Y.at(i));
        } else {
            yTest.push_back(Y.at(i));
        }
    }

    tensor X_train(
        {static_cast<int64>(trainSize), static_cast<int64>(featureCount)},
        xTrain.data(),
        dev
    );

    tensor Y_train(
        {static_cast<int64>(trainSize), 1},
        yTrain.data(),
        dev
    );

    tensor X_test(
        {static_cast<int64>(testSize), static_cast<int64>(featureCount)},
        xTest.data(),
        dev
    );

    tensor Y_test(
        {static_cast<int64>(testSize), 1},
        yTest.data(),
        dev
    );

    return {
        X_train,
        Y_train,
        X_test,
        Y_test
    };
}

void DataFrame::parseCsv(const std::string &path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("CSV file could not open.");
    }

    std::string line;

    if (std::getline(file, line)) {

        std::stringstream ss(line);
        std::string col;

        while (std::getline(ss, col, ',')) {

            this->m_columns.push_back(col);
            this->m_data[col] = {};
        }
    }

    while (std::getline(file, line)) {

        std::stringstream ss(line);
        std::string value;

        size_t colIdx = 0;

        while (std::getline(ss, value, ',')) {

            float32 v = std::stof(value);

            this->m_data[this->m_columns[colIdx]].push_back(v);

            colIdx++;
        }

        this->m_rows++;
    }
}
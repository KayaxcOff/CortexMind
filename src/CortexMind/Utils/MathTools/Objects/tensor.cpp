//
// Created by muham on 8.11.2025.
//

#include "CortexMind/Utils/MathTools/Object/tensor.hpp"
#include <iostream>

using namespace cortex::vecs;

MindTensor::MindTensor(const size_t _rows, const size_t _cols, const double initialValue) {
    this->rows = _rows;
    this->cols = _cols;
    this->tensor.resize(_cols, std::vector(_rows, initialValue));
}

const std::vector<std::vector<double>>& MindTensor::data() const {
    return this->tensor;
}

std::vector<std::vector<double>>& MindTensor::data() {
    return this->tensor;
}

void MindTensor::fill() {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            this->tensor[i][j] = 0.0;
}

void MindTensor::print() const {
    for (const auto& row : tensor) {
        for (const auto& val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

MindTensor MindTensor::transpose() const {
    MindTensor output(this->rows, this->cols, 0.0);
    for (size_t i = 0; i < this->rows; ++i)
        for (size_t j = 0; j < this->cols; ++j)
            output(j, i) = this->tensor[i][j];
    return output;
}

void MindTensor::check_shape(const MindTensor &other) const {
    if (rows != other.get_rows() || cols != other.get_cols())
        throw std::invalid_argument("Tensor2D: shape mismatch for operation");
}
//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_TENSOR_HPP
#define CORTEXMIND_TENSOR_HPP

#include <vector>
#include <stdexcept>

namespace cortex::vecs {
    class MindTensor {
    public:
        MindTensor(size_t _rows, size_t _cols, double initialValue = 0.0);

        MindTensor(const MindTensor&) = default;
        MindTensor(MindTensor&&) noexcept = default;
        MindTensor& operator=(const MindTensor&) = default;
        MindTensor& operator=(MindTensor&&) noexcept = default;

        [[nodiscard]] size_t get_rows() const { return rows; }
        [[nodiscard]] size_t get_cols() const { return cols; }

        double& operator()(const size_t i, const size_t j) {
            if (i >= rows || j >= cols)
                throw std::out_of_range("Tensor2D: index out of range");
            return this->tensor[i][j];
        }

        const double& operator()(const size_t i, const size_t j) const {
            if (i >= rows || j >= cols)
                throw std::out_of_range("Tensor2D: index out of range");
            return this->tensor[i][j];
        }

        [[nodiscard]] const std::vector<std::vector<double>>& data() const;
        std::vector<std::vector<double>>& data();

        void fill();
        void print() const;

        MindTensor operator+(const MindTensor& other) const {
            check_shape(other);
            MindTensor result(rows, cols);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = tensor[i][j] + other(i, j);
            return result;
        }

        MindTensor operator-(const MindTensor& other) const {
            check_shape(other);
            MindTensor result(rows, cols);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = tensor[i][j] - other(i, j);
            return result;
        }

        MindTensor operator*(const double scalar) const {
            MindTensor result(rows, cols);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result(i, j) = tensor[i][j] * scalar;
            return result;
        }

        MindTensor operator+=(const MindTensor& other) {
            check_shape(other);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    tensor[i][j] += other(i, j);
            return *this;
        }

        [[nodiscard]] MindTensor transpose() const;
    private:
        std::vector<std::vector<double>> tensor;
        size_t rows, cols;

        void check_shape(const MindTensor& other) const;
    };
}

#endif

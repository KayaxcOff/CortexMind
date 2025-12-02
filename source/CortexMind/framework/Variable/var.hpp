//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_VAR_HPP
#define CORTEXMIND_VAR_HPP

#include <vector>
#include <array>

namespace cortex::fw {
    class MindTensor {
    public:
        explicit MindTensor(size_t _batch,size_t _row, size_t _col, bool _grad=false);
        MindTensor(const MindTensor& other);
        MindTensor& operator=(const MindTensor& other);
        ~MindTensor();

        double& operator()(size_t b, size_t r, size_t c);
        double operator()(size_t b, size_t r, size_t c) const;

        MindTensor operator+(const MindTensor& other) const;
        MindTensor operator-(const MindTensor& other) const;
        MindTensor operator*(const MindTensor& other) const;
        MindTensor operator/(const MindTensor& other) const;

        MindTensor operator+(double scalar) const;
        MindTensor operator-(double scalar) const;
        MindTensor operator*(double scalar) const;
        MindTensor operator/(double scalar) const;

        MindTensor& operator+=(const MindTensor& other);
        MindTensor& operator-=(const MindTensor& other);
        MindTensor& operator*=(const MindTensor& other);
        MindTensor& operator/=(const MindTensor& other);

        MindTensor& operator+=(double scalar);
        MindTensor& operator-=(double scalar);
        MindTensor& operator*=(double scalar);
        MindTensor& operator/=(double scalar);

        void zero();
        void fill(double _value);
        void uniform_rand(double min, double max);
        void print() const;

        [[nodiscard]] std::vector<size_t> get_shape() const {return this->shape;}
        [[nodiscard]] std::vector<size_t> get_shape() {return this->shape;}
        [[nodiscard]] std::vector<size_t> get_strides() const {return this->strides;}
        [[nodiscard]] std::vector<size_t> get_strides() {return this->strides;}
        [[nodiscard]] bool isRequiresGrad() const {return this->required_grad;}
        [[nodiscard]] std::vector<double> get_data() const {return this->data;}
        [[nodiscard]] std::vector<double>& get_data() { return this->data; }

        [[nodiscard]] bool is_empty() const {return this->data.empty();}

        [[nodiscard]] MindTensor matmul(const MindTensor& other) const;
        [[nodiscard]] MindTensor transpose() const;
        [[nodiscard]] MindTensor reshape(const std::vector<size_t>& new_shape) const;
    private:
        std::vector<size_t> shape;
        std::vector<size_t> strides;
        std::vector<double> data;

        bool required_grad;

        [[nodiscard]] size_t getIdx(size_t batch, size_t row, size_t col) const;
    };
}

#endif //CORTEXMIND_VAR_HPP
//
// Created by muham on 3.11.2025.
//

#include "CortexMind/Utils/MathTools/vector/vector.hpp"

#include <iostream>
#include <cmath>

using namespace cortex::math;

MindVector MindVector::operator+(const MindVector& other) const {
    MindVector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = _data_[i] + other[i];
    }
    return result;
}

MindVector MindVector::operator-(const MindVector& other) const {
    MindVector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = _data_[i] - other[i];
    }
    return result;
}

MindVector MindVector::operator*(const double scalar) const {
    MindVector result = *this;
    for (auto& val : result) {
        val *= scalar;
    }
    return result;
}

MindVector& MindVector::operator+=(const MindVector& other) {
    for (size_t i = 0; i < size(); ++i) {
        _data_[i] += other[i];
    }
    return *this;
}

MindVector& MindVector::operator-=(const MindVector& other) {
    for (size_t i = 0; i < size(); ++i) {
        _data_[i] -= other[i];
    }
    return *this;
}

MindVector& MindVector::operator*=(const double scalar) {
    for (size_t i = 0; i < size(); ++i) {
        _data_[i] *= scalar;
    }
    return *this;
}

double MindVector::dot(const MindVector& other) const {
    double result = 0.0;
    for (size_t i = 0; i < size(); ++i) {
        result += _data_[i] * other[i];
    }
    return result;
}

double MindVector::norm() const {
    double sum = 0.0;
    for (const auto& val : _data_) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

void MindVector::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << ": ";
    }
    std::cout << "[";
    for (size_t i = 0; i < size(); ++i) {
        std::cout << _data_[i];
        if (i < size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
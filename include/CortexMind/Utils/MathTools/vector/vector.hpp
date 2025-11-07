//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_VECTOR_HPP
#define CORTEXMIND_VECTOR_HPP

#include <vector>
#include <string>

namespace cortex::math {
    class MindVector {
    public:
        MindVector() = default;
        explicit MindVector(const size_t size, const double value = 0.0) : _data_(size, value) {}
        MindVector(const std::initializer_list<double>init) : _data_(init) {}
        explicit MindVector(const std::vector<double>& init) : _data_(init) {}

        // size of vector
        [[nodiscard]] size_t size() const { return _data_.size(); }

        // access operator
        double& operator[](const size_t index) { return _data_[index]; }
        const double& operator[](const size_t index) const { return _data_[index]; }

        auto begin() { return _data_.begin(); }
        auto end() { return _data_.end(); }
        [[nodiscard]] auto begin() const { return _data_.begin(); }
        [[nodiscard]] auto end() const { return _data_.end(); }

        // vector operations
        MindVector operator+(const MindVector& other) const;
        MindVector operator-(const MindVector& other) const;
        MindVector operator*(double scalar) const;
        MindVector& operator+=(const MindVector& other);
        MindVector& operator-=(const MindVector& other);
        MindVector& operator*=(double scalar);

        // helpful funcs
        [[nodiscard]] double dot(const MindVector& other) const;
        [[nodiscard]] double norm() const;
        void print(const std::string& name = "") const;

        // resize vector
        void resize(const size_t newSize, const double value = 0.0) {
            _data_.resize(newSize, value);
        }

        // clear vector
        void clear() {
            _data_.clear();
        }

        // convert to std::vector
        [[nodiscard]] std::vector<double> to_vector() const {
            return _data_;
        }
        std::vector<double>& to_vector() {
            return _data_;
        }
    private:
        std::vector<double> _data_;
    };
}

#endif //CORTEXMIND_VECTOR_HPP
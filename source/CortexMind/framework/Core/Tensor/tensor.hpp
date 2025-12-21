//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_TENSOR_HPP
#define CORTEXMIND_TENSOR_HPP

#include <CortexMind/framework/Core/Array/array.hpp>
#include <vector>
#include <array>

namespace cortex::_fw {
    class MindTensor {
    public:
        explicit MindTensor(int batch = 0, int channel = 0, int height = 0, int width = 0, float value=0.0f);
        MindTensor(const MindTensor& other);
        ~MindTensor();

        [[nodiscard]] float& at(int b, int c, int h, int w) noexcept;
        [[nodiscard]] const float& at(int b, int c, int h, int w) const noexcept;
        [[nodiscard]] std::vector<AlignedArray<float, 8>>& data() noexcept;
        [[nodiscard]] std::array<int, 4> shape() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] AlignedArray<float, 8>& dataIdx(size_t idx) noexcept;
        [[nodiscard]] int batch() const noexcept;
        [[nodiscard]] int channel() const noexcept;
        [[nodiscard]] int height() const noexcept;
        [[nodiscard]] int width() const noexcept;
        [[nodiscard]] size_t vec_size() const;
        [[nodiscard]] float* raw_ptr(size_t idx);
        [[nodiscard]] const float *raw_ptr(size_t idx) const noexcept;

        void print() const noexcept;
        void uniform_rand(float lower=0.0f, float upper=1.0f) noexcept;
        void zero() noexcept;
        void fill(float value) noexcept;
        void allocate(int batch, int channel, int height, int width) noexcept;

        [[nodiscard]] MindTensor flatten() const noexcept;
        [[nodiscard]] MindTensor matmul(const MindTensor& other) const noexcept;
        [[nodiscard]] MindTensor transpose() const noexcept;
        [[nodiscard]] MindTensor permute(std::array<int, 4> axes) const noexcept;

        void operator()(int b, int c, int h, int w) noexcept;

        MindTensor& operator=(const MindTensor& other);

        MindTensor operator+(const MindTensor& other) const noexcept;
        MindTensor operator-(const MindTensor& other) const noexcept;
        MindTensor operator*(const MindTensor& other) const noexcept;
        MindTensor operator/(const MindTensor& other) const noexcept;

        MindTensor operator+(float scalar) const noexcept;
        MindTensor operator-(float scalar) const noexcept;
        MindTensor operator*(float scalar) const noexcept;
        MindTensor operator/(float scalar) const noexcept;

        MindTensor& operator+=(const MindTensor& other) noexcept;
        MindTensor& operator-=(const MindTensor& other) noexcept;
        MindTensor& operator*=(const MindTensor& other) noexcept;
        MindTensor& operator/=(const MindTensor& other) noexcept;
    private:
        std::vector<AlignedArray<float, 8>> m_data;
        std::array<int, 4> m_shape;
        size_t m_size;
    };
}

#endif //CORTEXMIND_TENSOR_HPP
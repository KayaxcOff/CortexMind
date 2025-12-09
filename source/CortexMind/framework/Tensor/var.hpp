//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_VAR_HPP
#define CORTEXMIND_VAR_HPP

#include <CortexMind/framework/Array/array.hpp>
#include <vector>
#include <array>
#include <ostream>

namespace cortex::_fw {
    /**
     * @brief Represents a 4-dimensional tensor, the fundamental data structure in CortexMind.
     * * MindTensor stores data in a vector of AlignedArray<float, 8> blocks to enable
     * highly optimized Single Instruction, Multiple Data (SIMD) operations using AVX2.
     * The standard shape is (Batch, Channels, Height, Width).
     */
    class MindTensor {
    public:

        /**
         * @brief Constructs a new MindTensor object.
         * * @param batch The size of the batch dimension (N).
         * @param channels The number of channels (C).
         * @param height The height dimension (H).
         * @param width The width dimension (W).
         * @param initValue The initial value to fill the tensor elements with.
         */
        explicit MindTensor(int batch=0, int channels=0, int height=0, int width=0, float initValue=0.0f);

        /**
         * @brief Default destructor.
         */
        ~MindTensor() = default;

        /**
         * @brief Copy constructor. Performs a deep copy of the tensor data.
         * * @param other The tensor to be copied.
         */
        MindTensor(const MindTensor &other);

        // --- Element Access ---

        /**
         * @brief Provides a writable reference to the element at the specified 4D index.
         * * Calculates the flat index and returns the reference from the AlignedArray block.
         * @param b Batch index.
         * @param c Channel index.
         * @param h Height index.
         * @param w Width index.
         * @return float& Writable reference to the element.
         */
        float& at(int b, int c, int h, int w);

        /**
         * @brief Provides a read-only reference to the element at the specified 4D index.
         * @param b Batch index.
         * @param c Channel index.
         * @param h Height index.
         * @param w Width index.
         * @return const float& Read-only reference to the element.
         */
        [[nodiscard]] const float& at(int b, int c, int h, int w) const;


        // --- Properties ---

        /**
         * @brief Returns the total number of elements in the tensor (B * C * H * W).
         * @return size_t Total size of the underlying data.
         */
        [[nodiscard]] size_t size() const { return this->m_size; }

        /**
         * @brief Returns the vector of AlignedArray blocks storing the data.
         * @return std::vector<AlignedArray<float, 8>> The data container.
         */
        [[nodiscard]] std::vector<AlignedArray<float, 8>> data() const { return this->m_data; }

        /**
         * @brief Returns a specific AlignedArray block (8 elements) by index.
         * @param idx Index of the block in the data vector.
         * @return AlignedArray<float, 8> The requested data block.
         */
        [[nodiscard]] AlignedArray<float, 8> dataIdx(const size_t idx) const { return this->m_data[idx]; }

        /**
         * @brief Performs a sum across the same offset index of all AlignedArray blocks.
         * * NOTE: This method has limited practical use in standard DL operations.
         * @param idx The index (0-7) within each block to sum.
         * @return float The resulting sum.
         */
        [[nodiscard]] float alignedIdx(size_t idx) const;

        /**
         * @brief Returns the shape of the tensor as an array (B, C, H, W).
         * @return std::array<int, 4> The shape array.
         */
        [[nodiscard]] std::array<int, 4> shape() const { return this->m_shape; }

        /**
         * @brief Returns the size of a specific dimension.
         * @param idx The dimension index (0=Batch, 1=Channel, 2=Height, 3=Width).
         * @return int The size of the dimension.
         */
        [[nodiscard]] int shapeIdx(const int idx) const { return this->m_shape[idx]; }

        /**
         * @brief Prints the tensor shape and elements to the console for debugging.
         */
        void print();

        // --- Initialization and Utility ---

        /**
         * @brief Fills the tensor with random numbers from a uniform distribution.
         * @param min Minimum value (default -1.0f).
         * @param max Maximum value (default 1.0f).
         */
        void uniform_rand(float min=-1.0f, float max=1.0f);

        /**
         * @brief Sets all elements of the tensor to zero (0.0f).
         */
        void zero();

        /**
         * @brief Sets all elements of the tensor to a specific scalar value.
         * @param value The scalar value.
         */
        void fill(float value);

        /**
         * @brief Returns a new tensor that is a flattened view of the original.
         * * The shape is transformed from (B, C, H, W) to (B, C*H*W, 1, 1).
         * @return MindTensor The flattened tensor.
         */
        [[nodiscard]] MindTensor flatten() const;

        /**
         * @brief Performs matrix multiplication (Dot Product) C = A * B.
         * * Assumes A is flattened to (B, N, 1, 1) and B is a weight matrix (N, M, 1, 1).
         * Uses AVX optimized matmul_kernel.
         * @param other The right-hand side tensor (B).
         * @return MindTensor The resulting tensor (C).
         */
        [[nodiscard]] MindTensor matmul(const MindTensor &other) const;

        /**
         * @brief Performs a transpose operation (permute {0, 2, 3, 1} typically).
         * @return MindTensor The transposed tensor.
         */
        [[nodiscard]] MindTensor transpose() const;

        /**
         * @brief Returns a new tensor with dimensions reordered according to the given axes.
         * * @param axes An array defining the new order of axes (e.g., {0, 3, 1, 2}).
         * @return MindTensor The permuted tensor.
         */
        [[nodiscard]] MindTensor permute(std::array<int, 4> axes) const;


        // --- Operator Overloads (Assignment) ---

        /**
         * @brief Stream output operator for printing the tensor.
         * @param os The output stream.
         * @param tensor The tensor to print.
         * @return std::ostream& Reference to the output stream.
         */
        friend std::ostream& operator<<(std::ostream& os, const MindTensor& tensor) {
            const std::vector shape(tensor.m_shape.begin(), tensor.m_shape.end());
            std::vector indices(4, 0);
            TensorRecursive(tensor, shape, indices, 0, 0, os);
            return os;
        }

        /**
         * @brief Assignment operator. Copies the shape and data using SIMD load/store.
         * @param other The tensor to assign from.
         * @return MindTensor& Reference to the resulting tensor.
         */
        MindTensor& operator=(const MindTensor& other);


        // --- Operator Overloads (Element-wise Arithmetic: Tensor-Tensor) ---

        /**
         * @brief Element-wise addition of two tensors (A + B).
         */
        MindTensor operator+(const MindTensor& other) const;

        /**
         * @brief Element-wise subtraction of two tensors (A - B).
         */
        MindTensor operator-(const MindTensor& other) const;

        /**
         * @brief Element-wise multiplication of two tensors (A * B).
         */
        MindTensor operator*(const MindTensor& other) const;

        /**
         * @brief Element-wise division of two tensors (A / B).
         */
        MindTensor operator/(const MindTensor& other) const;


        // --- Operator Overloads (Compound Assignment: Tensor-Tensor) ---

        /**
         * @brief Compound assignment: A += B. Uses AVX add_kernel.
         */
        MindTensor& operator+=(const MindTensor& other);

        /**
         * @brief Compound assignment: A -= B. Uses AVX sub_kernel.
         */
        MindTensor& operator-=(const MindTensor& other);

        /**
         * @brief Compound assignment: A *= B. Uses AVX mul_kernel.
         */
        MindTensor& operator*=(const MindTensor& other);

        /**
         * @brief Compound assignment: A /= B. Uses AVX div_kernel.
         */
        MindTensor& operator/=(const MindTensor& other);


        // --- Operator Overloads (Element-wise Arithmetic: Tensor-Scalar) ---

        /**
         * @brief Element-wise addition with a scalar (A + scalar).
         */
        MindTensor operator+(float scalar) const;

        /**
         * @brief Element-wise subtraction with a scalar (A - scalar).
         */
        MindTensor operator-(float scalar) const;

        /**
         * @brief Element-wise multiplication with a scalar (A * scalar).
         */
        MindTensor operator*(float scalar) const;

        /**
         * @brief Element-wise division with a scalar (A / scalar).
         */
        MindTensor operator/(float scalar) const;


        // --- Operator Overloads (Compound Assignment: Tensor-Scalar) ---

        /**
         * @brief Compound assignment: A += scalar. Uses AVX add_kernel with broadcast scalar.
         */
        MindTensor& operator+=(float scalar);

        /**
         * @brief Compound assignment: A -= scalar. Uses AVX sub_kernel with broadcast scalar.
         */
        MindTensor& operator-=(float scalar);

        /**
         * @brief Compound assignment: A *= scalar. Uses AVX mul_kernel with broadcast scalar.
         */
        MindTensor& operator*=(float scalar);

        /**
         * @brief Compound assignment: A /= scalar. Uses AVX div_kernel with broadcast scalar.
         */
        MindTensor& operator/=(float scalar);
    private:
        std::vector<AlignedArray<float, 8>> m_data; ///< Underlying data stored as AVX-aligned blocks.
        std::array<int, 4> m_shape; ///< The tensor dimensions (B, C, H, W).
        size_t m_size; ///< Total number of elements (B*C*H*W).

        /**
         * @brief Helper function to get the raw pointer to the start of the AlignedArray block
         * that corresponds to the given flat index.
         * * This is CRITICAL for safe, high-performance SIMD memory access.
         * @param flat_idx The flat (linear) index of the element.
         * @return float* Pointer to the first element of the corresponding AlignedArray block.
         */
        [[nodiscard]] float* getIdx(size_t flat_idx);

        static void TensorRecursive(const MindTensor &tensor, const std::vector<int> &shape, std::vector<int> &indices, int dim, int indent, std::ostream &os);
    };
}

#endif //CORTEXMIND_VAR_HPP
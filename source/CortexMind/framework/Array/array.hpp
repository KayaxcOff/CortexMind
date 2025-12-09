//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_ARRAY_HPP
#define CORTEXMIND_ARRAY_HPP

#include <CortexMind/framework/AVX2/avx2.hpp>
#include <CortexMind/framework/ErrorSystem/debug.hpp>
#include <type_traits>
#include <immintrin.h>

namespace cortex::_fw {
    /**
     * @brief A fixed-size array template optimized for SIMD operations.
     * * This class ensures that the underlying data storage is aligned in memory,
     * which is a requirement for high-performance SIMD instructions like AVX2.
     * In CortexMind, this is primarily used as AlignedArray<float, 8> (a 32-byte block)
     * to hold exactly one __m256 register's worth of data.
     * * @tparam T The type of elements stored (expected to be float32).
     * @tparam N The number of elements (expected to be 8 for AVX2 float support).
     * @tparam Alignment The memory alignment boundary (expected to be 32 bytes).
     */
    template <typename T, std::size_t N, std::size_t Alignment = 32>
    class AlignedArray {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable for SIMD usage.");
    public:
        using value_type = T;
        static constexpr std::size_t size_value = N;
        static constexpr std::size_t alignment_value = Alignment;
        /**
         * @brief Default constructor. Elements are zero-initialized.
         */
        AlignedArray() = default;
        /**
         * @brief Constructs the array and initializes all elements to a specific value.
         * @param initialValue The value to fill the array with.
         */
        explicit AlignedArray(const T& initialValue) {
            this->fill(initialValue);
        }
        /**
         * @brief Default destructor.
         */
        ~AlignedArray() = default;
        // --- Pointers and Accessors ---

        /**
         * @brief Returns a pointer to the first element of the aligned data block.
         * * This pointer is guaranteed to meet the alignment requirement.
         */
        [[nodiscard]] T* data() noexcept {return this->m_data;}
        [[nodiscard]] const T* data() const noexcept {return this->m_data;}
        /**
         * @brief Checks if the array is empty (based on m_active, though N is fixed).
         */
        [[nodiscard]] bool empty() const noexcept {return this->m_active == 0;}
        /**
         * @brief Returns the number of active elements (N, which is typically 8).
         */
        [[nodiscard]] constexpr std::size_t size() const noexcept {return this->m_active;}
        /**
         * @brief Returns a pointer to the element at the specified index.
         */
        [[nodiscard]] T* idx(std::size_t index) noexcept {return &this->m_data[index];}
        [[nodiscard]] const T* idx(std::size_t index) const noexcept {return &this->m_data[index];}
        /**
         * @brief Returns a reference to the element at the specified index with bounds checking.
         */
        [[nodiscard]] T& at(std::size_t index) {
            if (index >= N) SynapticNode::captureFault(true, "cortex::_fw::AlignedArray::at()", "index out of bounds");
            return this->m_data[index];
        }
        [[nodiscard]] const T& at(std::size_t index) const {
            if (index >= N) SynapticNode::captureFault(true, "cortex::_fw::AlignedArray::at()", "index out of bounds");
            return this->m_data[index];
        }
        /**
         * @brief Returns an iterator to the beginning.
         */
        [[nodiscard]] T* begin() noexcept { return this->m_data; }
        [[nodiscard]] const T* begin() const noexcept { return this->m_data; }
        /**
         * @brief Returns an iterator to the end.
         */
        [[nodiscard]] T* end() noexcept { return this->m_data + N; }
        [[nodiscard]] const T* end() const noexcept { return this->m_data + N; }
        // --- SIMD Load/Store ---

        /**
         * @brief Performs a raw aligned load into an __m256 register (AVX intrinsic).
         * @return __m256 The loaded vector register.
         */
        [[nodiscard]] __m256 load() const noexcept {
            static_assert(std::is_same_v<T, float>, "AVX load() only supports float.");
            static_assert(N == 8, "AVX load() requires N = 8.");
            return _mm256_load_ps(reinterpret_cast<const float*>(this->m_data));
        }
        /**
         * @brief Calculates the horizontal sum of all elements in the block using SIMD reduction.
         * @return T The scalar sum of all elements.
         */
        [[nodiscard]] T sum() const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            return avx::h_sum(this->load_reg());
        }
        /**
         * @brief Finds the minimum value in the block using SIMD reduction.
         * @return T The scalar minimum value.
         */
        [[nodiscard]] T min() const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            return avx::min(this->load_reg());
        }
        /**
         * @brief Finds the maximum value in the block using SIMD reduction.
         * @return T The scalar maximum value.
         */
        [[nodiscard]] T max() const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            return avx::max(this->load_reg());
        }
        /**
         * @brief Calculates the mean (average) value of the block.
         * @return T The scalar mean.
         */
        [[nodiscard]] T mean() const noexcept {
            return static_cast<double>(this->sum()) / N;
        }
        /**
         * @brief Calculates the dot product (inner product) with another block.
         * * Uses AVX mul and horizontal sum (h_sum).
         * @param other The right-hand side array.
         * @return T The scalar dot product.
         */
        [[nodiscard]] T dot(const AlignedArray& other) const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            const avx::reg prod = avx::mul(this->load_reg(), other.load_reg());
            return avx::h_sum(prod);
        }
        /**
         * @brief Calculates the L2-norm (Euclidean norm) of the block.
         */
        [[nodiscard]] T norm() const noexcept {return static_cast<T>(std::sqrt(dot(*this)));}

        /**
         * @brief Sets all elements to their default initialized value (e.g., 0.0f).
         */
        void clear() noexcept {
            std::fill_n(this->m_data, N, T{});
        }

        /**
         * @brief Fills all elements of the array with a specific scalar value.
         * @param value The scalar value.
         */
        void fill(const T& value) noexcept {
            for (std::size_t i = 0; i < N; ++i) this->m_data[i] = value;
        }
        /**
         * @brief Performs a raw aligned store from an __m256 register (AVX intrinsic).
         * @param reg The vector register to store.
         */
        void store(const __m256 reg) const noexcept {
            static_assert(std::is_same_v<T, float>, "AVX store() only supports float.");
            static_assert(N == 8, "AVX store() requires N = 8.");
            _mm256_store_ps(reinterpret_cast<float*>(this->m_data), reg);
        }
        void store(const __m256 reg) noexcept {
            static_assert(std::is_same_v<T, float>, "AVX store() only supports float.");
            static_assert(N == 8, "AVX store() requires N = 8.");
            _mm256_store_ps(reinterpret_cast<float*>(this->m_data), reg);
        }

        /**
         * @brief Standard array indexing operator.
         */
        [[nodiscard]] T& operator[](std::size_t index) noexcept {return this->m_data[index];}
        [[nodiscard]] const T& operator[](std::size_t index) const noexcept {return this->m_data[index];}

        /**
         * @brief Loads the data block into an AVX register using the high-level avx::load interface.
         * @return __m256 The loaded AVX register.
         */
        [[nodiscard]] __m256 load_reg() const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8, "Only float[8] supported");
            return avx::load(this->m_data);
        }

        /**
         * @brief Stores the AVX register back into the data block using the high-level avx::store interface.
         * @param v The AVX register to store.
         */
        void store_reg(__m256 v) noexcept {
            static_assert(std::is_same_v<T, float> && N == 8, "Only float[8] supported");
            avx::store(this->m_data, v);
        }

        /**
         * @brief Attempts to resize the 'active' portion of the array.
         * * NOTE: Since N is fixed, this only adjusts m_active and is generally only used for tail elements.
         * @param new_size The new active size (must be <= N).
         * @param value The value to fill new active elements with.
         */
        void resize(size_t new_size, float value) noexcept {
            if (N > 8) {
                SynapticNode::captureFault(true, "cortex::_fw::AlignedArray::resize()", "N > 8 is not supported for AVX2 operations");
                return;
            }
            if (new_size > N) new_size = N;
            this->m_active = new_size;
            for (size_t i = 0; i < new_size; ++i) {
                this->m_data[i] = value;
            }
        }

        /**
         * @brief Element-wise addition using AVX. Returns a new array.
         */
        AlignedArray operator+(const AlignedArray& rhs) const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            AlignedArray result;
            result.store_reg(avx::add(this->load_reg(), rhs.load_reg()));
            return result;
        }

        /**
         * @brief Element-wise subtraction using AVX. Returns a new array.
         */
        AlignedArray operator-(const AlignedArray& rhs) const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            AlignedArray result;
            result.store_reg(avx::sub(this->load_reg(), rhs.load_reg()));
            return result;
        }

        /**
         * @brief Element-wise multiplication using AVX. Returns a new array.
         */
        AlignedArray operator*(const AlignedArray& rhs) const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            AlignedArray result;
            result.store_reg(avx::mul(this->load_reg(), rhs.load_reg()));
            return result;
        }

        /**
         * @brief Element-wise division using AVX. Returns a new array.
         */
        AlignedArray operator/(const AlignedArray& rhs) const noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            AlignedArray result;
            result.store_reg(avx::div(this->load_reg(), rhs.load_reg()));
            return result;
        }

        // --- Operator Overloads (SIMD Compound Assignment) ---

        /**
         * @brief Compound assignment (+=) using AVX.
         */
        AlignedArray& operator+=(const AlignedArray& rhs) noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            store_reg(avx::add(this->load_reg(), rhs.load_reg()));
            return *this;
        }

        /**
         * @brief Compound assignment (-=) using AVX.
         */
        AlignedArray& operator-=(const AlignedArray& rhs) noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            store_reg(avx::sub(this->load_reg(), rhs.load_reg()));
            return *this;
        }

        /**
         * @brief Compound assignment (*=) using AVX.
         */
        AlignedArray& operator*=(const AlignedArray& rhs) noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            store_reg(avx::mul(this->load_reg(), rhs.load_reg()));
            return *this;
        }

        /**
         * @brief Compound assignment (/=) using AVX.
         */
        AlignedArray& operator/=(const AlignedArray& rhs) noexcept {
            static_assert(std::is_same_v<T, float> && N == 8);
            store_reg(avx::div(this->load_reg(), rhs.load_reg()));
            return *this;
        }

    private:
        alignas(Alignment) T m_data[N]{}; ///< The aligned data buffer.
        size_t m_active = N; ///< The number of currently active elements.
    };
}

#endif //CORTEXMIND_ARRAY_HPP
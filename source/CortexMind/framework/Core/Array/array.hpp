//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_ARRAY_HPP
#define CORTEXMIND_ARRAY_HPP

#include <algorithm>

namespace cortex::_fw {
    template <typename T, std::size_t N, std::size_t Alignment = 32>
    class AlignedArray {
    public:
        AlignedArray() = default;
        ~AlignedArray() = default;

        [[nodiscard]] T* ptr() noexcept                 {return this->alg_data;}
        [[nodiscard]] const T* ptr() const noexcept     {return this->alg_data;}
        [[nodiscard]] std::size_t size() const noexcept {return N;}
        [[nodiscard]] bool is_aligned() const noexcept  {return reinterpret_cast<std::uintptr_t>(this->alg_data) % Alignment == 0;}
        [[nodiscard]] bool empty() const noexcept       {return N == 0;}

        void fill(const T& value) noexcept {
            for (std::size_t i = 0; i < N; ++i) this->alg_data[i] = value;
        }
        void clear() noexcept {
            std::fill_n(this->alg_data, N, T{});
        }

        T& operator[](std::size_t idx) noexcept             { return this->alg_data[idx]; }
        const T& operator[](std::size_t idx) const noexcept { return this->alg_data[idx]; }

    private:
        alignas(Alignment) T alg_data[N]{};
    };
}

#endif //CORTEXMIND_ARRAY_HPP
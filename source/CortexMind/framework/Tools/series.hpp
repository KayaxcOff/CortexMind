//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP

#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/framework/Tools/dtype.hpp>
#include <CortexMind/framework/Tools/types.hpp>
#include <string>
#include <variant>
#include <vector>

namespace cortex::_fw {
    using SeriesVariant = std::variant<std::vector<f32>, std::vector<bool>, std::vector<std::string>>;

    /**
     * @brief 1D heterogeneous data series with type safety.
     *
     * A flexible container that can hold `float32`, `bool`, or `std::string` data.
     * Provides type-safe access via `as<T>()` and basic preprocessing for numeric data.
     */
    class Series {
    public:
        /**
         * @brief Constructs a Series from float32 data.
         */
        explicit Series(std::vector<f32> data = {});
        /**
         * @brief Constructs a Series from boolean data.
         */
        explicit Series(std::vector<bool> data);
        /**
         * @brief Constructs a Series from string data.
         */
        explicit Series(std::vector<std::string> data);
        ~Series();

        /**
         * @brief Returns the data type of the series.
         */
        [[nodiscard]]
        DType dtype() const;

        /**
         * @brief Returns a reference to the underlying vector of the specified type.
         *
         * @tparam T Must match the actual stored type (`f32`, `bool`, or `std::string`)
         * @throws Assertion failure if type mismatch
         */
        template<typename T>
        std::vector<T>& as() {
            auto* ptr = std::get_if<std::vector<T>>(&this->m_data);
            CXM_ASSERT(!ptr, "Dtype mismatch");

            return *ptr;
        }

        /**
         * @brief Returns a const reference to the underlying vector of the specified type.
         *
         * @tparam T Must match the actual stored type (`f32`, `bool`, or `std::string`)
         * @throws Assertion failure if type mismatch
         */
        template<typename T>
        const std::vector<T>& as() const {
            const auto* ptr = std::get_if<std::vector<T>>(&this->m_data);
            CXM_ASSERT(!ptr, "Dtype mismatch");
            return *ptr;
        }

        /**
         * @brief Normalizes the series by dividing all elements by a given value.
         *
         * Only supported for `float32` series.
         *
         * @param value Non-zero divisor
         */
        void normalize(f32 value);
        /**
         * @brief Performs min-max scaling to range [0, 1].
         *
         * Only supported for `float32` series.
         */
        void scale();

        /**
         * @brief Returns the number of elements in the series.
         */
        [[nodiscard]]
        size_t size() const;
        /**
         * @brief Checks if the series is empty.
         */
        [[nodiscard]]
        bool empty() const;

        /**
         * @brief Returns mutable reference to float32 data (only for float32 series).
         */
        [[nodiscard]]
        std::vector<f32>& data();
        /**
         * @brief Returns const reference to float32 data (only for float32 series).
         */
        [[nodiscard]]
        const std::vector<f32>& data() const;

        f32 operator[](size_t index) const;
    private:
        SeriesVariant m_data;
        DType m_dtype;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP
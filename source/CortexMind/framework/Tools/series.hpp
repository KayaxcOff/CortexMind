//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <vector>

namespace cortex::_fw {
    /**
     * @brief 1D data series container with basic preprocessing utilities.
     *
     * Provides a lightweight wrapper around `std::vector<f32>` with common
     * operations such as normalization, scaling (min-max), and safe access.
     */
    class Series {
    public:
        /**
         * @brief Constructs a Series from a vector of values.
         *
         * @param data Initial data (moved into the series)
         */
        explicit Series(std::vector<f32> data = {});
        ~Series();

        /**
         * @brief Normalizes the series by dividing all elements by a given value.
         *
         * @param value Value to divide by (must be non-zero)
         */
        void normalize(f32 value);
        /**
         * @brief Performs min-max scaling to range [0, 1].
         *
         * If all values are equal, no scaling is applied.
         */
        void scale();
        /**
         * @brief Returns a mutable reference to the underlying data vector.
         */
        [[nodiscard]]
        std::vector<f32>& data();
        /**
         * @brief Returns a const reference to the underlying data vector.
         */
        [[nodiscard]]
        const std::vector<f32>& data() const;
        /**
         * @brief Access element at given index with bounds checking.
         *
         * @param index Index of the element
         * @return Value at the specified index
         */
        f32 operator[](size_t index) const;
    private:
        std::vector<f32> m_data;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP
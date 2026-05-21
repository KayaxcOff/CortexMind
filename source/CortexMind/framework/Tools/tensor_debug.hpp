//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_TENSOR_DEBUG_HPP
#define CORTEXMIND_TENSOR_DEBUG_HPP

#include <CortexMind/framework/Tensor/tensor.hpp>

namespace cortex::_fw {
    /**
     * @brief Static utility class for debugging and inspecting tensors.
     *
     * Contains convenient methods for logging tensor information, validating
     * gradients for anomalies (NaN, Inf, exploding/vanishing), and formatting
     * tensor metadata.
     */
    class TensorDebug {
    public:
        /**
         * @brief Returns a string representation of the tensor shape.
         *
         * @param t Tensor to inspect
         * @return Shape as formatted string (e.g. "(32 64 28 28)")
         */
        static std::string shape_str(const Tensor& t);
        /**
         * @brief Returns basic statistical information about the tensor.
         *
         * Includes max, min, and mean values in scientific notation.
         *
         * @param t Tensor to analyze
         * @return Statistics string (e.g. "max=1.23e+00 min=-0.45e+00 mean=0.12e+00")
         */
        static std::string stats_str(const Tensor& t);
        /**
         * @brief Validates gradient tensor for common training problems.
         *
         * Checks for:
         * - NaN values
         * - Infinite values (±Inf)
         * - Exploding gradients (|value| > 1e8)
         * - Vanishing gradients (|value| < 1e-20 and != 0)
         *
         * Logs appropriate warnings/errors using the framework Logger.
         *
         * @param grad        Gradient tensor to validate
         * @param tensor_name Name of the tensor (for logging)
         */
        static void validateGradient(const Tensor& grad, const std::string& tensor_name);
        /**
         * @brief Logs tensor information (shape and optional statistics).
         *
         * @param name       Name/identifier for the tensor
         * @param t          Tensor to log
         * @param show_stats Whether to include statistical information (max/min/mean)
         */
        static void logTensor(const std::string& name, const Tensor& t, bool show_stats = true);
    };
} // namespace cortex::debug

#endif //CORTEXMIND_TENSOR_DEBUG_HPP
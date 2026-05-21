//
// Created by muham on 21.05.2026.
//

#ifndef CORTEXMIND_TENSOR_DEBUG_HPP
#define CORTEXMIND_TENSOR_DEBUG_HPP

#include <CortexMind/framework/Tensor/tensor.hpp>
#include <CortexMind/framework/Tools/logger.hpp>
#include <sstream>

namespace cortex::_fw {

class TensorDebug {
public:
    static std::string shape_str(const _fw::Tensor& t) {
        std::string result = "(";
        const auto& shape = t.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            result += std::to_string(shape[i]);
            if (i < shape.size() - 1) result += ", ";
        }
        result += ")";
        return result;
    }

    static std::string stats_str(const _fw::Tensor& t) {
        if (t.len() == 0) return "empty";

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3);
        oss << "max=" << t.max()
            << " min=" << t.min()
            << " mean=" << t.mean();
        return oss.str();
    }

    static void validateGradient(const _fw::Tensor& grad,
                                 const std::string& tensor_name) {
        if (grad.len() == 0) return;

        float max_val = grad.max();
        if (isNaN(max_val)) {
            Logger::getInstance().error(
                "NaN detected in " + tensor_name + " gradient!");
        } else if (isInf(max_val)) {
            Logger::getInstance().error(
                "Inf detected in " + tensor_name + " gradient!");
        } else if (std::abs(max_val) > 1e8f) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(3) << max_val;
            Logger::getInstance().warn(
                tensor_name + " gradient extremely large: " + oss.str());
        } else if (std::abs(max_val) < 1e-20f && max_val != 0.0f) {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(3) << max_val;
            Logger::getInstance().warn(
                tensor_name + " gradient vanishing: " + oss.str());
        }
    }

    static void logTensor(const std::string& name,
                         const _fw::Tensor& t,
                         const bool show_stats = true) {
        std::string msg = name + " shape=" + shape_str(t);
        if (show_stats) {
            msg += " " + stats_str(t);
        }
        Logger::getInstance().debug(msg);
    }
};

} // namespace cortex::debug

#endif //CORTEXMIND_TENSOR_DEBUG_HPP
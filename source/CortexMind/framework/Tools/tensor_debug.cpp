//
// Created by muham on 21.05.2026.
//

#include "CortexMind/framework/Tools/tensor_debug.hpp"
#include <CortexMind/framework/Tools/logger.hpp>
#include <sstream>

using namespace cortex::_fw;

std::string TensorDebug::shape_str(const Tensor &t) {
    std::string output = "(";
    for (const auto& item : t.shape()) {
        output += std::to_string(item) + " ";
    }
    output += ")";
    return output;
}

std::string TensorDebug::stats_str(const Tensor &t) {
    if (t.len() == 0) {
        return "Tensor is empty";
    }

    std::ostringstream output;
    output << std::scientific << std::setprecision(3);
    output << "max=" << t.max() << " min=" << t.min() << " mean=" << t.mean();
    return output.str();
}

void TensorDebug::validateGradient(const Tensor &grad, const std::string &tensor_name) {
    validateTensor(grad, tensor_name, true);
}

bool TensorDebug::validateTensor(const Tensor &t, const std::string &tensor_name, const bool is_gradient) {
    if (t.len() == 0) return true;

    const float* ptr = t.get();
    const size_t length = t.len();

    bool has_nan = false;
    bool has_inf = false;
    float extreme_val = 0.0f;

    for (size_t i = 0; i < length; ++i) {
        if (std::isnan(ptr[i])) {
            has_nan = true;
            break;
        }
        if (std::isinf(ptr[i])) {
            has_inf = true;
            break;
        }
        if (std::abs(ptr[i]) > extreme_val) {
            extreme_val = std::abs(ptr[i]);
        }
    }

    const std::string type_str = is_gradient ? " gradient" : " activation";

    if (has_nan) {
        Logger::getInstance().error("[💥 IMPORTANT] NaN detected in " + tensor_name + type_str + "! Shape: " + shape_str(t));
        return false;
    }

    if (has_inf) {
        Logger::getInstance().error("[💥 IMPORTANT] Inf detected in " + tensor_name + type_str + "! Shape: " + shape_str(t));
        return false;
    }

    if (extreme_val > 1e8f) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3) << extreme_val;
        Logger::getInstance().warn(tensor_name + type_str + " extremely large (Exploding): " + oss.str());
    } else if (is_gradient && extreme_val < 1e-20f && extreme_val != 0.0f) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3) << extreme_val;
        Logger::getInstance().warn(tensor_name + type_str + " vanishing: " + oss.str());
    }

    return true;
}

void TensorDebug::logTensor(const std::string &name, const Tensor &t, const bool show_stats) {
    std::string msg = name + " shape=" + shape_str(t);
    if (show_stats) {
        msg += " " + stats_str(t);
    }
    Logger::getInstance().debug(msg);
}
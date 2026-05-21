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
    if (grad.len() == 0) {
        return;
    }

    if (const f32 max_val = grad.max(); isNaN(max_val)) {
        Logger::getInstance().error("NaN detected in " + tensor_name + " gradient!");
    } else if (isInf(max_val)) {
        Logger::getInstance().error("Inf detected in " + tensor_name + " gradient!");
    } else if (std::abs(max_val) > 1e8f) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3) << max_val;
        Logger::getInstance().warn(tensor_name + " gradient extremely large: " + oss.str());
    } else if (std::abs(max_val) < 1e-20f && max_val != 0.0f) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3) << max_val;
        Logger::getInstance().warn(tensor_name + " gradient vanishing: " + oss.str());
    }
}

void TensorDebug::logTensor(const std::string &name, const Tensor &t, const bool show_stats) {
    std::string msg = name + " shape=" + shape_str(t);
    if (show_stats) {
        msg += " " + stats_str(t);
    }
    Logger::getInstance().debug(msg);
}

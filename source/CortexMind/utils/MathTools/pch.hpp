//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_PCH_HPP
#define CORTEXMIND_PCH_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/framework/Log/log.hpp>
#include <stdexcept>

namespace cortex {
    static tensor add(const tensor& x, const tensor& y) {
        return x + y;
    }

    static tensor relu(const tensor& input) {
        auto output = input;
        for (double & i : output.get_data()) {
            if (i < 0.0) {
                i = 0.0;
            }
        }
        return output;
    }

    static tensor relu_prime(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get_data().size(); ++i) {
            output.get_data()[i] = (input.get_data()[i] > 0.0) ? 1.0 : 0.0;
        }
        return output;
    }

    static tensor sigmoid(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get_data().size(); ++i) {
            output.get_data()[i] = 1.0 / (1.0 + std::exp(-input.get_data()[i]));
        }
        return output;
    }

    static tensor tanh(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get_data().size(); ++i) {
            output.get_data()[i] = std::tanh(input.get_data()[i]);
        }
        return output;
    }

    static tensor logarithm(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get_data().size(); ++i) {
            if (output.get_data()[i] < 0.0) {
                log("logarithm: input is negative");
                throw std::runtime_error("logarithm: input is negative");
            }
            output.get_data()[i] = std::log(input.get_data()[i]);
        }
        return output;
    }

    static tensor exp(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get_data().size(); ++i) {
            output.get_data()[i] = std::exp(input.get_data()[i]);
        }
        return output;
    }

    static double mean(const tensor& input) {
        if (input.get_data().empty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (const double& val : input.get_data()) {
            sum += val;
        }
        return sum / static_cast<double>(input.get_data().size());
    }

    static double variance(const tensor& input, const double mean_val) {
        if (input.get_data().empty() || input.get_data().size() == 1) {
            return 0.0;
        }

        double sum = 0.0;

        for (const double& val : input.get_data()) {
            const double diff = val - mean_val;
            sum += diff * diff;
        }

        return sum / static_cast<double>(input.get_data().size());
    }
}

#endif //CORTEXMIND_PCH_HPP
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
        for (double & i : output.get()) {
            if (i < 0.0) {
                i = 0.0;
            }
        }
        return output;
    }

    static tensor relu_prime(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get().size(); ++i) {
            output.get()[i] = (input.get()[i] > 0.0) ? 1.0 : 0.0;
        }
        return output;
    }

    static tensor sigmoid(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get().size(); ++i) {
            output.get()[i] = 1.0 / (1.0 + std::exp(-input.get()[i]));
        }
        return output;
    }

    static tensor tanh(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get().size(); ++i) {
            output.get()[i] = std::tanh(input.get()[i]);
        }
        return output;
    }

    static tensor logarithm(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get().size(); ++i) {
            if (output.get()[i] < 0.0) {
                log("logarithm: input is negative");
                throw std::runtime_error("logarithm: input is negative");
            }
            output.get()[i] = std::log(input.get()[i]);
        }
        return output;
    }

    static tensor exp(const tensor& input) {
        tensor output = input;
        for (size_t i = 0; i < output.get().size(); ++i) {
            output.get()[i] = std::exp(input.get()[i]);
        }
        return output;
    }

    static double mean(const tensor& input) {
        if (input.get().empty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (const double& val : input.get()) {
            sum += val;
        }
        return sum / static_cast<double>(input.get().size());
    }

    static double variance(const tensor& input, const double mean_val) {
        if (input.get().empty() || input.get().size() == 1) {
            return 0.0;
        }

        double sum = 0.0;

        for (const double& val : input.get()) {
            const double diff = val - mean_val;
            sum += diff * diff;
        }

        return sum / static_cast<double>(input.get().size());
    }
}

#endif //CORTEXMIND_PCH_HPP
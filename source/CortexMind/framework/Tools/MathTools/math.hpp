//
// Created by muham on 10.12.2025.
//

#ifndef CORTEXMIND_MATH_HPP
#define CORTEXMIND_MATH_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::_fw {
    class TensorFn {
    public:
        TensorFn() = default;
        ~TensorFn() = default;

        static void relu(tensor& input);
        static void sigmoid(tensor& input);
        static void tanh(tensor& input);
        static void leaky_relu(tensor& input);
        static void softmax(tensor& input);
        static tensor mean(tensor& input);
        static tensor variance(tensor& input);
    };
}

#endif //CORTEXMIND_MATH_HPP
//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_CORTEXMIND_HPP
#define CORTEXMIND_CORTEXMIND_HPP

// ----- dataset -----
#include <CortexMind/dataset/circle.hpp>

// ----- net -----
#include <CortexMind/net/LossFunction/mae.hpp>
#include <CortexMind/net/LossFunction/mse.hpp>
#include <CortexMind/net/NeuralNetwork/convolution_2d.hpp>
#include <CortexMind/net/NeuralNetwork/dense.hpp>
#include <CortexMind/net/NeuralNetwork/flatten.hpp>
#include <CortexMind/net/NeuralNetwork/gelu.hpp>
#include <CortexMind/net/NeuralNetwork/gelu_exact.hpp>
#include <CortexMind/net/NeuralNetwork/input.hpp>
#include <CortexMind/net/NeuralNetwork/leaky_relu.hpp>
#include <CortexMind/net/NeuralNetwork/relu.hpp>
#include <CortexMind/net/NeuralNetwork/sigmoid.hpp>
#include <CortexMind/net/NeuralNetwork/sigmoid_fast.hpp>
#include <CortexMind/net/NeuralNetwork/silu.hpp>
#include <CortexMind/net/NeuralNetwork/silu_fast.hpp>
#include <CortexMind/net/NeuralNetwork/tanh.hpp>
#include <CortexMind/net/OptimizationFunction/sgd.hpp>

// ----- tools -----
#include <CortexMind/tools/math.hpp>
#include <CortexMind/tools/println.hpp>
#include <CortexMind/tools/tensor_meta.hpp>
#include <CortexMind/tools/types.hpp>
#include <CortexMind/tools/values.hpp>
#include <CortexMind/tools/version.hpp>

// ------ utility -----
#include <CortexMind/utility/cast.hpp>

#endif //CORTEXMIND_CORTEXMIND_HPP
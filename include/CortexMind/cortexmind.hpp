//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_CORTEXMIND_HPP
#define CORTEXMIND_CORTEXMIND_HPP

// ----- net -----
#include <CortexMind/net/LossFunction/mse.hpp>
#include <CortexMind/net/NeuralNetwork/dense.hpp>
#include <CortexMind/net/NeuralNetwork/flatten.hpp>
#include <CortexMind/net/NeuralNetwork/relu.hpp>
#include <CortexMind/net/OptimizationFunction/sgd.hpp>

// ----- tools -----
#include <CortexMind/tools/println.hpp>
#include <CortexMind/tools/tensor_meta.hpp>
#include <CortexMind/tools/types.hpp>
#include <CortexMind/tools/values.hpp>
#include <CortexMind/tools/version.hpp>

#endif //CORTEXMIND_CORTEXMIND_HPP
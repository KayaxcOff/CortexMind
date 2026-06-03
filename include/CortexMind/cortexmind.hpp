//
// Created by muham on 9.05.2026.
//

#ifndef CORTEXMIND_CORTEXMIND_HPP
#define CORTEXMIND_CORTEXMIND_HPP

// ----- dataset -----
#include <CortexMind/dataset/ProcessedDataset/circle.hpp>
#include <CortexMind/dataset/ProcessedDataset/spiral.hpp>
#include <CortexMind/dataset/ProcessedDataset/two_moons.hpp>

// ----- net -----
#include <CortexMind/net/LossFunction/bce.hpp>
#include <CortexMind/net/LossFunction/cce.hpp>
#include <CortexMind/net/LossFunction/cce_with_logit.hpp>
#include <CortexMind/net/LossFunction/mae.hpp>
#include <CortexMind/net/LossFunction/mse.hpp>
#include <CortexMind/net/Model/model.hpp>
#include <CortexMind/net/Metrics/acc.hpp>
#include <CortexMind/net/Metrics/approx.hpp>
#include <CortexMind/net/Metrics/mse.hpp>
#include <CortexMind/net/Metrics/rmse.hpp>
#include <CortexMind/net/NeuralNetwork/batch_norm.hpp>
#include <CortexMind/net/NeuralNetwork/convolution_2d.hpp>
#include <CortexMind/net/NeuralNetwork/dense.hpp>
#include <CortexMind/net/NeuralNetwork/dropout.hpp>
#include <CortexMind/net/NeuralNetwork/flatten.hpp>
#include <CortexMind/net/NeuralNetwork/gelu.hpp>
#include <CortexMind/net/NeuralNetwork/gelu_exact.hpp>
#include <CortexMind/net/NeuralNetwork/global_avg_pool_2d.hpp>
#include <CortexMind/net/NeuralNetwork/input.hpp>
#include <CortexMind/net/NeuralNetwork/leaky_relu.hpp>
#include <CortexMind/net/NeuralNetwork/relu.hpp>
#include <CortexMind/net/NeuralNetwork/sigmoid.hpp>
#include <CortexMind/net/NeuralNetwork/sigmoid_fast.hpp>
#include <CortexMind/net/NeuralNetwork/silu.hpp>
#include <CortexMind/net/NeuralNetwork/silu_fast.hpp>
#include <CortexMind/net/NeuralNetwork/softmax.hpp>
#include <CortexMind/net/NeuralNetwork/tanh.hpp>
#include <CortexMind/net/OptimizationFunction/adam.hpp>
#include <CortexMind/net/OptimizationFunction/momentum.hpp>
#include <CortexMind/net/OptimizationFunction/sgd.hpp>

// ----- tools -----
#include <CortexMind/tools/load.hpp>
#include <CortexMind/tools/math.hpp>
#include <CortexMind/tools/println.hpp>
#include <CortexMind/tools/tensor_meta.hpp>
#include <CortexMind/tools/types.hpp>
#include <CortexMind/tools/values.hpp>
#include <CortexMind/tools/version.hpp>

// ------ utility -----
#include <CortexMind/utility/Cast/factory.hpp>
#include <CortexMind/utility/DataFrame/frame.hpp>
#include <CortexMind/utility/Image/vm.hpp>

#endif //CORTEXMIND_CORTEXMIND_HPP
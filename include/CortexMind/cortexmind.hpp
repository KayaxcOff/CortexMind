//
// Created by muham on 21.02.2026.
//

#ifndef CORTEXMIND_CORTEXMIND_HPP
#define CORTEXMIND_CORTEXMIND_HPP

// <---------- Utils ---------->
#include <CortexMind/utils/Cast/factory.hpp>
#include <CortexMind/utils/Data/image.hpp>
#include <CortexMind/utils/Data/file.hpp>
#include <CortexMind/utils/Metric/metric.hpp>

// <---------- Dataset --------->
#include <CortexMind/dataset/Frame/df.hpp>
#include <CortexMind/dataset/Manager/data.hpp>

// <---------- Net ---------->
#include <CortexMind/net/NeuralNetwork/dense.hpp>
#include <CortexMind/net/NeuralNetwork/conv2d.hpp>
#include <CortexMind/net/NeuralNetwork/flatten.hpp>
#include <CortexMind/net/NeuralNetwork/relu.hpp>
#include <CortexMind/net/NeuralNetwork/input.hpp>
#include <CortexMind/net/NeuralNetwork/global_avg.hpp>
#include <CortexMind/net/NeuralNetwork/tanh.hpp>
#include <CortexMind/net/NeuralNetwork/sigmoid.hpp>
#include <CortexMind/net/NeuralNetwork/dropout.hpp>
#include <CortexMind/net/NeuralNetwork/batch_norm.hpp>
#include <CortexMind/net/NeuralNetwork/leaky_relu.hpp>
#include <CortexMind/net/LossFunctions/mse.hpp>
#include <CortexMind/net/LossFunctions/mae.hpp>
#include <CortexMind/net/LossFunctions/bce.hpp>
#include <CortexMind/net/OptimizationFunctions/sgd.hpp>
#include <CortexMind/net/OptimizationFunctions/adam.hpp>
#include <CortexMind/net/Model/model.hpp>
#include <CortexMind/net/Callbacks/early_stop.hpp>

// <---------- Tools ---------->
#include <CortexMind/tools/println.hpp>
#include <CortexMind/tools/version.hpp>
#include <CortexMind/tools/params.hpp>
#include <CortexMind/tools/devices.hpp>
#include <CortexMind/tools/cast.hpp>
#include <CortexMind/tools/cpp_version.hpp>
#include <CortexMind/tools/device.hpp>
#include <CortexMind/tools/defaults.hpp>

#endif //CORTEXMIND_CORTEXMIND_HPP
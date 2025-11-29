//
// Created by muham on 8.11.2025.
//

#include "CortexMind/DataPrep/data.hpp"

using namespace cortex::prep;

DataGen::DataGen() = default;
DataGen::~DataGen() = default;

std::vector<cortex::tensor> DataGen::float_to_tensor(const std::vector<float32>& data) {
    this->float_data_ = data;

    std::vector<tensor> tensors;

    for (float & i : this->float_data_) {
        tensors.emplace_back(i, i);
    }
    return tensors;
}

std::vector<cortex::tensor> DataGen::int_to_tensor(const std::vector<int32>& data) {
    this->int_data_ = data;

    std::vector<tensor> tensors;

    for (int32 & i : this->int_data_) {
        tensors.emplace_back(i, i);
    }
    return tensors;
}
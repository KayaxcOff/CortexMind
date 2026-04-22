//
// Created by muham on 21.04.2026.
//

#include "CortexMind/framework/Net/layer.hpp"
#include <type_traits>

using namespace cortex::_fw;

LayerBase::LayerBase(std::string name, const boolean _train_flag) : kName(std::move(name)), kTrainFlag(_train_flag) {}

LayerBase::~LayerBase() = default;

void LayerBase::TrainMode() {
    this->kTrainFlag = true;
}

void LayerBase::EvalMode() {
    this->kTrainFlag = false;
}

const std::string &LayerBase::name() const {
    return this->kName;
}
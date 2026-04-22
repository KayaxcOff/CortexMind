//
// Created by muham on 21.04.2026.
//

#include "CortexMind/framework/Net/layer.hpp"
#include <type_traits>

using namespace cortex::_fw;

LayerBase::LayerBase(std::string name, const boolean _train_flag) : m_name(std::move(name)), train_flag(_train_flag) {}

LayerBase::~LayerBase() = default;

void LayerBase::TrainMode() {
    this->train_flag = true;
}

void LayerBase::EvalMode() {
    this->train_flag = false;
}

const std::string &LayerBase::name() const {
    return this->m_name;
}
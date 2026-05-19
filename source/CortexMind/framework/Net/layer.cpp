//
// Created by muham on 19.05.2026.
//

#include "CortexMind/framework/Net/layer.hpp"

using namespace cortex::_fw;

LayerBase::LayerBase(std::string name, const bool _train_flag) : m_name(std::move(name)), m_train_flag(_train_flag) {}

LayerBase::~LayerBase() = default;

const std::string &LayerBase::name() const {
    return this->m_name;
}

bool LayerBase::flag() const {
    return this->m_train_flag;
}

void LayerBase::TrainMode() {
    this->m_train_flag = true;
}

void LayerBase::EvalMode() {
    this->m_train_flag = false;
}
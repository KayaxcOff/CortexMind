//
// Created by muham on 18.03.2026.
//

#include "CortexMind/core/Net/layer.hpp"
#include <type_traits>

using namespace cortex::_fw;
using namespace cortex;

Layer::Layer(const bool train, std::string name) : name(std::move(name)), train_flag(train) {}

Layer::Layer(Layer &&) noexcept = default;

Layer::~Layer() = default;

const std::string& Layer::config() const {
    return this->name;
}

void Layer::toEval() {
    this->train_flag = false;
}

void Layer::toTrain() {
    this->train_flag = true;
}

bool Layer::is_training() const noexcept {
    return this->train_flag;
}

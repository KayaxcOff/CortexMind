//
// Created by muham on 5.03.2026.
//


#include "CortexMind/core/Net/layer.hpp"
#include <utility>

using namespace cortex::_fw;
using namespace cortex;

Layer::Layer(const bool _train_flag, string info) : flag(_train_flag), info(std::move(info)) {}

bool Layer::is_train() const {
    return this->flag;
}

void Layer::set_train(const bool _train) {
    this->flag = _train;
}

string Layer::config() {
    return this->info;
}

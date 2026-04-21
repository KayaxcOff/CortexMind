//
// Created by muham on 21.04.2026.
//

#include "CortexMind/framework/Net/loss.hpp"
#include <utility>

using namespace cortex::_fw;

LossBase::LossBase(std::string name) : kName(std::move(name)) {}

LossBase::~LossBase() = default;

const std::string &LossBase::name() {
    return this->kName;
}
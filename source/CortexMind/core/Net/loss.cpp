//
// Created by muham on 6.03.2026.
//

#include "CortexMind/core/Net/loss.hpp"

using namespace cortex::_fw;

Loss::Loss(const string &name) {
    this->name = name;
}

cortex::string Loss::config() const {
    return this->name;
}

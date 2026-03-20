//
// Created by muham on 18.03.2026.
//

#include "CortexMind/core/Net/loss.hpp"
#include <type_traits>

using namespace cortex::_fw;
using namespace cortex;

Loss::Loss(std::string  name) : name(std::move(name)) {}

Loss::Loss(Loss &&) noexcept = default;

Loss::~Loss() = default;

const std::string &Loss::config() const {
    return this->name;
}

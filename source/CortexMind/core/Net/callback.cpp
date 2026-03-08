//
// Created by muham on 8.03.2026.
//

#include "CortexMind/core/Net/callback.hpp"
#include <utility>

using namespace cortex::_fw;
using namespace cortex;

Callback::Callback(string name) : name(std::move(name)) {}

string Callback::config() const {
    return this->name;
}

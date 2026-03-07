//
// Created by muham on 2.03.2026.
//

#include "CortexMind/core/Tools/throw.hpp"

using namespace cortex::_fw;

status::status(const bool status, const string &msg) : runtime_error(msg) {
    if (!status) throw *this;
}

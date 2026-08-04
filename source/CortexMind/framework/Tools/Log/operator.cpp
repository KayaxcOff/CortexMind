//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Tools/Log/w.hpp"

using namespace cortex::_fw;

WLog &WLog::operator<<(std::ostream &(*manip)(std::ostream &)) {
    manip(this->m_os);
    return *this;
}
//
// Created by muham on 4.08.2026.
//

#include "CortexMind/framework/Tools/Log/w.hpp"
#include <CortexMind/framework/Tools/Log/as_string.hpp>

using namespace cortex::_fw;

namespace {
    constexpr char kReset[] = "\033[0m";
} //unnamed namespace

WLog::WLog(const LogLevel level, std::ostream &os) : m_os(os) {
    this->m_os << ansi(level);
}

WLog::~WLog() noexcept {
    this->m_os << kReset;
}
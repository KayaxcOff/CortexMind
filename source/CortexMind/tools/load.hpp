//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_TOOLS_LOAD_HPP
#define CORTEXMIND_TOOLS_LOAD_HPP

#include <CortexMind/utility/DataFrame/frame.hpp>
#include <string>

namespace cortex {
    utils::DataFrame load(const std::string& file_name);
} //namespace cortex

#endif //CORTEXMIND_TOOLS_LOAD_HPP
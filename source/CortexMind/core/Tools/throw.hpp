//
// Created by muham on 2.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_THROW_HPP
#define CORTEXMIND_CORE_TOOLS_THROW_HPP

#include <CortexMind/tools/params.hpp>
#include <stdexcept>

namespace cortex::_fw {
    class status : public std::runtime_error {
    public:
        status(bool status, const string& msg);
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_THROW_HPP
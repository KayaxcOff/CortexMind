//
// Created by muham on 2.02.2026.
//

#ifndef CORTEXMIND_CORE_ENGINE_ERROR_HPP
#define CORTEXMIND_CORE_ENGINE_ERROR_HPP

#include <CortexMind/core/Tools/params.hpp>
#include <iostream>

namespace cortex::_fw {
    struct err {
        static void IsStatusOk(const bool status, const string &name, const string &msg) {
            if (!status) {
                std::cerr << "Error at " << name << std::endl;
                std::cerr << msg << std::endl;
                std::exit(1);
            }
        }
    };
} // namespace cortex::_fw

#define CXM_ASSERT(status, name, msg) \
(::cortex::_fw::err::IsStatusOk((status), (name), (msg)))

#endif //CORTEXMIND_CORE_ENGINE_ERROR_HPP
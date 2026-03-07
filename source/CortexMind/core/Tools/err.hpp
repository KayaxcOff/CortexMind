//
// Created by muham on 21.02.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_ERR_HPP
#define CORTEXMIND_CORE_TOOLS_ERR_HPP

#include <string>

namespace cortex::_fw {
    struct err {
        static void IsStatusOk(bool status, const std::string &name, const std::string &msg);
    };
} // namespace cortex::_fw

#define CXM_ASSERT(status, name, msg) \
(::cortex::_fw::err::IsStatusOk((status), (name), (msg)))

#endif //CORTEXMIND_CORE_TOOLS_ERR_HPP
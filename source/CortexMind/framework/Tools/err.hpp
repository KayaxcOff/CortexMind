//
// Created by muham on 31.03.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP

#include <string>

namespace cortex::_fw {
    struct err {
        static void exitIf(bool status, const std::string& message, const std::string& where);
        static void warnIf(bool status, const std::string& message, const std::string& where);
    };
} //namespace cortex::_fw

#define CXM_ASSERT(status, msg, whr) \
    (::cortex::_fw::err::exitIf((status), (msg), (whr)))

#define CXM_WARN(status, msg, whr) \
    (::cortex::_fw::err::warnIf((status), (msg), (whr)))

#endif //CORTEXMIND_FRAMEWORK_TOOLS_ERR_HPP
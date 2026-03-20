//
// Created by muham on 13.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_ERR_HPP
#define CORTEXMIND_CORE_TOOLS_ERR_HPP

#include <string>

namespace cortex::_fw {
    /**
     * @brief   Lightweight error reporting utilities
     *
     * Contains static methods that conditionally print messages to stderr
     * and optionally terminate the program.
     */
    struct err {
        /**
         * @brief   If status == false, prints message and terminates program with exit(1)
         * @param   status    Condition that must be true to avoid error
         * @param   address   Usually function name, file:line or context identifier
         * @param   message   Human-readable explanation of what went wrong
         *
         * @note    Message format on stderr:  address: message\n
         */
        static void exitIf(bool status, const std::string& address, const std::string& message);
        /**
         * @brief   If status == false, prints warning message to stderr
         *          (program execution continues)
         * @param   status    Condition that should ideally be true
         * @param   address   Context / location identifier
         * @param   message   Warning explanation
         *
         * @note    Does **not** terminate the program
         */
        static void warnIf(bool status, const std::string& address, const std::string& message);
    };
} // namespace cortex::_fw

/**
 * @def     CXM_ASSERT(status, whr, msg)
 * @brief   Fatal assertion: if !(status) → print and exit(1)
 * @param   status   Expression that must evaluate to true
 * @param   whr      Location/context string (usually __func__ or "module::func")
 * @param   msg      Error message
 *
 * @note    Condition is inverted compared to standard assert()
 */
#define CXM_ASSERT(status, whr, msg) \
    (::cortex::_fw::err::exitIf((status), (whr), (msg)))

/**
 * @def     CXM_WARN(status, whr, msg)
 * @brief   Non-fatal warning: if !(status) → print to stderr
 * @param   status   Expression that should preferably be true
 * @param   whr      Location/context string
 * @param   msg      Warning message
 */
#define CXM_WARN(status, whr, msg) \
    (::cortex::_fw::err::warnIf((status), (whr), (msg)))

#endif //CORTEXMIND_CORE_TOOLS_ERR_HPP
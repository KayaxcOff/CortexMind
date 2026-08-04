//
// Created by muham on 4.08.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_CONSOLE_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_CONSOLE_HPP

namespace cortex::_fw::detail {
    /**
     * @brief Enables ANSI escape sequence processing for the console.
     *
     * On Windows, this function enables virtual terminal processing
     * so that ANSI escape sequences used for colored output are
     * interpreted correctly.
     *
     * On platforms with native ANSI support, the function performs
     * no operation.
     */
    void EnableVirtualTerminal();
} //namespace cortex::_fw::detail

/// Initializes the console for colored output.
#define CXM_INITIALIZE_CONSOLE() \
    ::cortex::_fw::detail::EnableVirtualTerminal()

#endif //CORTEXMIND_FRAMEWORK_TOOLS_CONSOLE_HPP
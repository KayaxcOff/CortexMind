#ifndef CORTEXMIND_TOOLS_LOG_HPP
#define CORTEXMIND_TOOLS_LOG_HPP

#include <iostream>
#include <string>

namespace cortex {
	// @brief Logs a message to the standard output.
	inline void log(const std::string& message) {
		std::cout << message << std::endl;
	}
} // namespace cortex

#endif // CORTEXMIND_TOOLS_LOG_HPP
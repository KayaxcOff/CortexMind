//
// Created by muham on 6.02.2026.
//

#ifndef CORTEXMIND_TOOLS_VERSION_HPP
#define CORTEXMIND_TOOLS_VERSION_HPP

/**
 * @def CXM_MAJOR
 * @brief Major version number of the CortexMind library.
 *
 * Increased when incompatible API changes are introduced.
 */
#define CXM_MAJOR 3
/**
 * @def CXM_MINOR
 * @brief Minor version number of the CortexMind library.
 *
 * Increased when functionality is added in a backward-compatible manner.
 */

#define CXM_MINOR 0
/**
 * @def CXM_PATCH
 * @brief Patch version number of the CortexMind library.
 *
 * Increased when backward-compatible bug fixes are applied.
 */
#define CXM_PATCH 0

#include <iostream>

namespace cortex {
    /**
     * @brief Prints the current CortexMind version to the standard output.
     *
     * Outputs the version in the format:
     * `CortexMind v<MAJOR>.<MINOR>.<PATCH>`
     *
     * @example
     * @code
     * cortex::PrintVersion();
     * // Output: CortexMind v3.0.0
     * @endcode
     */
    inline void PrintVersion() {
        std::cout << "CortexMind v" << CXM_MAJOR << "." << CXM_MINOR << "." << CXM_PATCH << std::endl;
    }
} // namespace cortex

#endif //CORTEXMIND_TOOLS_VERSION_HPP
//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_TOOLS_LOAD_HPP
#define CORTEXMIND_TOOLS_LOAD_HPP

#include <CortexMind/utility/DataFrame/frame.hpp>
#include <string>

namespace cortex {
    /**
     * @brief Loads a CSV file into a `DataFrame`.
     *
     * This is a convenient wrapper function that creates and returns a
     * `DataFrame` by reading the specified CSV file.
     *
     * The CSV file is expected to have a header row with column names.
     *
     * @param file_name Path to the CSV file
     * @return DataFrame containing the loaded tabular data
     *
     * @example
     * @code
     * auto df = load("data/iris.csv");
     * df.info();
     * df.head(10);
     * @endcode
     */
    utils::DataFrame load(const std::string& file_name);
} //namespace cortex

#endif //CORTEXMIND_TOOLS_LOAD_HPP
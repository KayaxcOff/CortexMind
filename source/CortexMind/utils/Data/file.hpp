//
// Created by muham on 2.03.2026.
//

#ifndef CORTEXMIND_UTILS_DATA_FILE_HPP
#define CORTEXMIND_UTILS_DATA_FILE_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex::utils {
    class FileNet {
    public:
        FileNet();
        ~FileNet();

        static tensor loadFromCSV(const string& path, bool header = false, bool _requires_grad = false);
        static tensor loadFromJSON(const string& path, bool _requires_grad = false);
    };
} // namespace cortex::utils

#endif //CORTEXMIND_UTILS_DATA_FILE_HPP
//
// Created by muham on 24.04.2026.
//

#ifndef CORTEXMIND_UTILS_DATA_FRAME_FRAME_HPP
#define CORTEXMIND_UTILS_DATA_FRAME_FRAME_HPP

#include <CortexMind/tools/params.hpp>
#include <string>
#include <variant>
#include <vector>

namespace cortex::utils {
    class DataFrame {
    public:
        explicit DataFrame(std::string path);
        ~DataFrame();

    private:
        std::variant<int32, float32, boolean, std::string> variables;
        std::vector<std::string> feats;
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILS_DATA_FRAME_FRAME_HPP
//
// Created by muham on 21.12.2025.
//

#ifndef CORTEXMIND_TEXT_HPP
#define CORTEXMIND_TEXT_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <iostream>

namespace cortex::tools {
    class TextVec {
    public:
        TextVec() = default;
        ~TextVec() = default;

        static tensor to_tensor(const std::string& path);
    };
}

#endif //CORTEXMIND_TEXT_HPP
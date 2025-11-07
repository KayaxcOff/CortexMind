//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_TOKENIZER_HPP
#define CORTEXMIND_TOKENIZER_HPP

#include <string>
#include <CortexMind/Utils/MathTools/vector/vector.hpp>

namespace cortex::tools {
    class MindTokenizer {
    public:
        explicit MindTokenizer();

        std::vector<math::MindVector> tokenize(const std::string &_input);
    private:
        std::string input;
        std::vector<math::MindVector> tokens;
    };
}

#endif //CORTEXMIND_TOKENIZER_HPP
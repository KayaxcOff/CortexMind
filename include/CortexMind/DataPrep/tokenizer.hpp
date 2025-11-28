//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_TOKENIZER_HPP
#define CORTEXMIND_TOKENIZER_HPP

#include <CortexMind/Utils/params.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>

namespace cortex::prep {
    class MindTokenizer {
    public:
        MindTokenizer();
        ~MindTokenizer();

        std::vector<std::string> tokenize(const std::string& input);
        tensor to_tensor(const std::string& input);
    private:
        std::vector<float64> encode(const std::string& token);
    };
}

#endif //CORTEXMIND_TOKENIZER_HPP
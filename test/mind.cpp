//
// Created by muham on 30.11.2025.
//

#include "../include/CortexMind/cortexmind.hpp"

using namespace cortex;

int main() {
    tools::MindTokenizer tokenizer;

    const std::vector<std::string> tokens = {"CortexMind is a Machine Learning library in C++"};

    tokenizer.fit(tokens);

    std::vector<int> result = tokenizer.tokenize("is");

    for (const auto& token : result) {
        std::cout << token << std::endl;
    }

    return 0;
}
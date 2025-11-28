//
// Created by muham on 8.11.2025.
//

#include "CortexMind/DataPrep/tokenizer.hpp"

using namespace cortex::prep;

MindTokenizer::MindTokenizer() = default;

MindTokenizer::~MindTokenizer() = default;

std::vector<std::string> MindTokenizer::tokenize(const std::string &input) {
    std::vector<std::string> tokens;

    std::stringstream ss(input);
    std::string token;
    while (ss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

cortex::tensor MindTokenizer::to_tensor(const std::string &input) {
    const std::vector<std::string> tokens = this->tokenize(input);
    std::vector<std::vector<float64>> encoded_tokens;

    for (const auto &token : tokens) {
        encoded_tokens.push_back(this->encode(token));
    }

    const size rows = encoded_tokens.size();
    const size cols = encoded_tokens.empty() ? 0 : encoded_tokens[0].size();

    tensor result(rows, cols);

    for (size i = 0; i < rows; ++i) {
        for (size j = 0; j < cols; ++j) {
            result(i, j) = encoded_tokens[i][j];
        }
    }

    return result;
}

std::vector<cortex::float64> MindTokenizer::encode(const std::string &token) {
    constexpr int dim = 10;
    std::vector vec(dim, 0.0);

    const size hash = std::hash<std::string>{}(token);
    vec[hash % dim] = 1.0;

    return vec;
}
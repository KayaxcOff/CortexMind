//
// Created by muham on 4.11.2025.
//

#include "CortexMind/Utils/TextData/tokenizer.hpp"

using namespace cortex::tools;
using namespace cortex;

MindTokenizer::MindTokenizer() {
    this->input = "";
    this->tokens = {};
}

std::vector<math::MindVector> MindTokenizer::tokenize(const std::string &_input) {
    this->tokens.clear();

    for (const char c : _input) {
        math::MindVector vec(1);
        vec[0] = static_cast<double>(c)/127.0;
        this->tokens.push_back(vec);
    }

    return this->tokens;
}


/*
std::vector<math::MindVector> MindTokenizer::tokenize(const std::string &_input) {
    this->input = _input;

    this->tokens.clear();
    this->tokens.resize(this->input.size());

    for (const char c : this->input) {
        this->tokens.push_back(static_cast<math::MindVector>(c));
    }
    return this->tokens;
}
*/

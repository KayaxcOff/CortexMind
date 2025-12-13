//
// Created by muham on 12.12.2025.
//

#include "CortexMind/framework/Tools/Views/Text/text.hpp"
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <iostream>
#include <fstream>

using namespace cortex::_fw;

TensorText::TensorText() : rowIdx(0), colIdx(0) {}

void TensorText::initialize(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        CXM_ASSERT(true, "Failed to open file: " + filename);
    }
    this->texts.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> words;
        std::string word;
        for (const char c : line) {
            if (c == ' ') {
                if (!word.empty()) {
                    words.push_back(word);
                    word.clear();
                }
            } else {
                word += c;
            }
        }
        if (!word.empty()) {
            words.push_back(word);
        }
        this->texts.push_back(words);
    }
    this->rowIdx = static_cast<int>(this->texts.size());
    this->colIdx = this->texts.empty() ? 0 : static_cast<int>(this->texts[0].size());

    file.close();
}
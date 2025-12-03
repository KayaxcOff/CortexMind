//
// Created by muham on 3.12.2025.
//

#include "CortexMind/tools/Text/token.hpp"
#include <algorithm>
#include <ranges>
#include <sstream>

using namespace cortex::tools;

MindTokenizer::MindTokenizer(const size_t _vocab_size) : nextIdx(0), vocab_size(_vocab_size) {
    this->PAD_ID = static_cast<int>(this->nextIdx);

    this->vocabulary[this->PAD_TOKEN] = this->PAD_ID;
    this->reverse_vocabulary[this->PAD_ID] = this->PAD_TOKEN;
    this->nextIdx++;

    this->UNK_ID = static_cast<int>(this->nextIdx);
    this->vocabulary[this->UNK_TOKEN] = this->UNK_ID;
    this->reverse_vocabulary[this->UNK_ID] = this->UNK_TOKEN;
    this->nextIdx++;

    this->vocabulary[this->EOS_TOKEN] = static_cast<int>(this->nextIdx);
    this->reverse_vocabulary[static_cast<int>(this->nextIdx)] = this->EOS_TOKEN;
    this->nextIdx++;
}

std::vector<std::string> MindTokenizer::preprocess(const std::string &text) {
    std::string processed_text = text;

    std::ranges::transform(processed_text, processed_text.begin(), [](const unsigned char c){ return std::tolower(c); });

    for (char& c : processed_text) {
        if (ispunct(c) && c != '\'') {
            c = ' ';
        }
    }

    std::vector<std::string> tokens;
    std::stringstream ss(processed_text);
    std::string token;

    while (ss >> token) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

void MindTokenizer::fit(const std::vector<std::string> &corpus) {
    std::map<std::string, size_t> word_freq;

    for (const std::string& text : corpus) {
        for (std::vector<std::string> tokens = preprocess(text); const std::string& token : tokens) {
            word_freq[token]++;
        }
    }

    std::vector<std::pair<std::string, size_t>> sorted_freq(word_freq.begin(), word_freq.end());
    std::ranges::sort(sorted_freq, [](const auto& a, const auto& b) {return a.second > b.second; });

    size_t words_added = 0;
    for (const auto &key: sorted_freq | std::views::keys) {
        const std::string& word = key;
        if (word == this->UNK_TOKEN || word == this->PAD_TOKEN || word == this->EOS_TOKEN) {
            continue;
        }

        if (words_added + this->nextIdx >= this->vocab_size) {
            break;
        }

        this->vocabulary[word] = static_cast<int>(this->nextIdx);
        this->reverse_vocabulary[static_cast<int>(this->nextIdx)] = word;
        this->nextIdx++;
        words_added++;
    }
}

std::vector<int> MindTokenizer::tokenize(const std::string& texts) const {
    const std::vector<std::string> tokens = preprocess(texts);
    std::vector<int> indices;

    for (const std::string& token : tokens) {
        if (auto it = this->vocabulary.find(token); it != this->vocabulary.end()) {
            indices.push_back(it->second);
        } else {
            indices.push_back(this->UNK_ID);
        }
    }
    return indices;
}

std::string MindTokenizer::decode(const std::vector<int>& indices) const {
    std::stringstream ss;
    for (int index : indices) {
        if (auto it = this->reverse_vocabulary.find(index); it != this->reverse_vocabulary.end()) {
            if (const std::string& word = it->second; word != this->PAD_TOKEN && word != this->UNK_TOKEN && word != this->EOS_TOKEN) {
                ss << word << " ";
            } else if (word == this->UNK_TOKEN) {
                ss << this->UNK_TOKEN << " ";
            }
        }
    }
    std::string result = ss.str();
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

int MindTokenizer::idx(const std::string& token) const {
    if (const auto it = this->vocabulary.find(token); it != this->vocabulary.end()) {
        return it->second;
    }
    return UNK_ID;
}

std::string MindTokenizer::token(const int idx) const {
    if (const auto it = this->reverse_vocabulary.find(idx); it != this->reverse_vocabulary.end()) {
        return it->second;
    }
    return "";
}
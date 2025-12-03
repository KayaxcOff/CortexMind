//
// Created by muham on 3.12.2025.
//

#ifndef CORTEXMIND_TOKEN_HPP
#define CORTEXMIND_TOKEN_HPP

#include <string>
#include <vector>
#include <map>

namespace cortex::tools {
    class MindTokenizer {
    public:
        explicit MindTokenizer(size_t _vocab_size = 200);
        ~MindTokenizer() = default;

        void fit(const std::vector<std::string>& corpus);

        [[nodiscard]] std::vector<int> tokenize(const std::string& texts) const;
        [[nodiscard]] std::string decode(const std::vector<int>& indices) const;
        [[nodiscard]] size_t size() const {return this->reverse_vocabulary.size();}
        [[nodiscard]] int idx(const std::string& token) const;
        [[nodiscard]] std::string token(int idx) const;

        int UNK_ID;
        int PAD_ID;
    private:
        std::map<int, std::string> reverse_vocabulary;
        std::map<std::string, int> vocabulary;

        size_t nextIdx;
        size_t vocab_size;

        const std::string UNK_TOKEN = "<UNK>";
        const std::string PAD_TOKEN = "<PAD>";
        const std::string EOS_TOKEN = "<EOS>";

        [[nodiscard]] static std::vector<std::string> preprocess(const std::string& text);
    };
}

#endif //CORTEXMIND_TOKEN_HPP
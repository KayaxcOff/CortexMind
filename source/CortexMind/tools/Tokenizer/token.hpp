//
// Created by muham on 13.12.2025.
//

#ifndef CORTEXMIND_TOKEN_HPP
#define CORTEXMIND_TOKEN_HPP

#include <string>
#include <unordered_map>

namespace cortex::tools {
    class TokenNet {
    public:
        static inline const std::string PAD_TOKEN = "<PAD>";
        static inline const std::string UNK_TOKEN = "<UNK>";
        static inline const std::string BOS_TOKEN = "<BOS>";
        static inline const std::string EOS_TOKEN = "<EOS>";

        static constexpr int PAD_ID = 0;
        static constexpr int UNK_ID = 1;
        static constexpr int BOS_ID = 2;
        static constexpr int EOS_ID = 3;

        TokenNet();
        ~TokenNet();

        void fit(const std::string& token);
        void fit(const std::vector<std::string>& tokens);
        std::vector<int> tokenize(const std::string& token, bool add_special_tokens);
        std::vector<int> encode(const std::string &text, int max_length, bool add_special_tokens, bool truncate);
        [[nodiscard]] int getId(const std::string& token) const;
        [[nodiscard]] std::string decode(const std::vector<int>& ids, bool skip_special_tokens) const;
        [[nodiscard]] std::string getToken(int id) const;
        [[nodiscard]] size_t vocab_size() const;
        [[nodiscard]] size_t size() const;
    private:
        std::unordered_map<std::string, int> tokensIdx;
        std::unordered_map<int, std::string> reverseIdx;
        int nextIdx;
    };
}

#endif //CORTEXMIND_TOKEN_HPP
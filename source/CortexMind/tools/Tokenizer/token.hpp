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
        static constexpr auto UNK_TOKEN = "<UNK>";
        static constexpr int UNK_ID = 0;

        TokenNet();
        ~TokenNet();

        void fit(const std::string& token);
        void fit(const std::vector<std::string>& tokens);
        std::vector<int> tokenize(const std::string& token);
        [[nodiscard]] int getId(const std::string& token) const;
        [[nodiscard]] std::string getToken(int id) const;
        [[nodiscard]] size_t size() const;
    private:
        std::unordered_map<std::string, int> tokensIdx;
        std::unordered_map<int, std::string> reverseIdx;
        int nextIdx;
    };
}

#endif //CORTEXMIND_TOKEN_HPP
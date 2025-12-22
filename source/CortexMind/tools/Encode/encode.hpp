//
// Created by muham on 13.12.2025.
//

#ifndef CORTEXMIND_ENCODE_HPP
#define CORTEXMIND_ENCODE_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/tools/Tokenizer/token.hpp>
#include <memory>

namespace cortex::tools {
    class MindTransform {
    public:
        explicit MindTransform(std::unique_ptr<TokenNet> tokenizer);
        ~MindTransform() = default;

        [[nodiscard]] tensor encode(const std::string &text, int max_seq_len, bool add_special_tokens, bool truncate) const;
        [[nodiscard]] tensor encode_batch(const std::vector<std::string> &texts, int max_seq_len, bool add_special_tokens, bool truncate) const;
        [[nodiscard]] std::vector<std::string> decode(const tensor &encoded, bool skip_special_tokens) const;
        [[nodiscard]] int vocab_size() const;
    private:
        std::unique_ptr<TokenNet> token_net_;
    };
}

#endif //CORTEXMIND_ENCODE_HPP
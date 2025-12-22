//
// Created by muham on 13.12.2025.
//

#include "CortexMind/tools/Encode/encode.hpp"

using namespace cortex::tools;
using namespace cortex;

MindTransform::MindTransform(std::unique_ptr<TokenNet> tokenizer) : token_net_(std::move(tokenizer)) {}


tensor MindTransform::encode(const std::string &text, const int max_seq_len, const bool add_special_tokens, const bool truncate) const {
    const std::vector<int> ids = this->token_net_->encode(text, max_seq_len, add_special_tokens, truncate);

    tensor output(1, 1, max_seq_len, 1, static_cast<float>(TokenNet::PAD_ID));

    for (size_t t = 0; t < ids.size() && t < static_cast<size_t>(max_seq_len); ++t) {
        output.at(0, 0, static_cast<int>(t), 0) = static_cast<float>(ids[t]);
    }

    return output;
}

tensor MindTransform::encode_batch(const std::vector<std::string> &texts, const int max_seq_len, const bool add_special_tokens, const bool truncate) const {
    const int batch_size = static_cast<int>(texts.size());

    tensor output(batch_size, 1, max_seq_len, 1, static_cast<float>(TokenNet::PAD_ID));

    for (int b = 0; b < batch_size; ++b) {
        const std::vector<int> ids = this->token_net_->encode(texts[b], max_seq_len,
                                                              add_special_tokens, truncate);

        for (size_t t = 0; t < ids.size() && t < static_cast<size_t>(max_seq_len); ++t) {
            output.at(b, 0, static_cast<int>(t), 0) = static_cast<float>(ids[t]);
        }
    }

    return output;
}

std::vector<std::string> MindTransform::decode(const tensor &encoded, const bool skip_special_tokens) const {
    const int batch_size = encoded.batch();
    const int seq_len = encoded.height();

    std::vector<std::string> results;
    results.reserve(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        std::vector<int> ids;
        ids.reserve(seq_len);

        for (int t = 0; t < seq_len; ++t) {
            ids.push_back(static_cast<int>(encoded.at(b, 0, t, 0)));
        }

        results.push_back(this->token_net_->decode(ids, skip_special_tokens));
    }

    return results;
}

int MindTransform::vocab_size() const {
    return static_cast<int>(this->token_net_->vocab_size());
}
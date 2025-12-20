//
// Created by muham on 13.12.2025.
//

#include "CortexMind/tools/Encode/encode.hpp"

using namespace cortex::tools;
using namespace cortex;

MindTransform::MindTransform(std::unique_ptr<TokenNet> tokenizer) : token_net_(std::move(tokenizer)) {}

tensor MindTransform::encode(const std::string &text, const int maxSeqLen, const int batch) const {
    const std::vector<int> ids = this->token_net_->tokenize(text);
    tensor output(batch, 1, maxSeqLen, 1, static_cast<float>(TokenNet::UNK_ID));

    const int limit = std::min(static_cast<int>(ids.size()), maxSeqLen);

    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < limit; ++t) {
            output.at(b, 0, t, 0) = static_cast<float>(ids[t]);
        }
    }

    return output;
}

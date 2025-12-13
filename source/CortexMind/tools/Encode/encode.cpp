//
// Created by muham on 13.12.2025.
//

#include "CortexMind/tools/Encode/encode.hpp"

using namespace cortex::tools;
using namespace cortex;

MindTransform::MindTransform(std::unique_ptr<TokenNet> tokenizer) : token_net_(std::move(tokenizer)) {}

tensor MindTransform::encode(const std::string &text, const int maxSeqLen, const int batch) const {
    const std::vector<int> ids = this->token_net_->tokenize(text);

    tensor output(batch, 1, maxSeqLen, 1);

    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < ids.size(); ++j) {
            output.at(i, 0, j, 0) = static_cast<float>(ids[j]);
        }
    }
    return output;
}

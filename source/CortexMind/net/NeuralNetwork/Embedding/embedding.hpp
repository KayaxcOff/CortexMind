//
// Created by muham on 14.12.2025.
//

#ifndef CORTEXMIND_EMBEDDING_HPP
#define CORTEXMIND_EMBEDDING_HPP

#include <CortexMind/framework/Net/layer.hpp>

namespace cortex::nn {
    class Embedding : _fw::Layer {
    public:
        Embedding(int _input_dim, int _vocab_size);
        ~Embedding() override;

        tensor forward(const tensor &input) override;
        tensor backward(const tensor &grad_output) override;
        std::string config() override;
    private:
        int input_dim;
        int vocab_size;
    };
}

#endif //CORTEXMIND_EMBEDDING_HPP
//
// Created by muham on 19.12.2025.
// Test: Dense + ReLU + MAE + Manual SGD
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    auto tok = std::make_unique<tools::TokenNet>();
    tok->fit({"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"});

    tools::MindTransform enc(std::move(tok));

    const std::string text = "the quick fox jumps high";
    constexpr int maxSeqLen = 5;

    auto input_tensor = enc.encode(text, maxSeqLen);
    log("Input tensor: ");
    input_tensor.print();

    constexpr int vocab_size = 10;
    constexpr int embedding_dim = 16;
    nn::Embedding embedding(embedding_dim, vocab_size);
    net::MeanSquared mse;
    net::Momentum optimizer(0.01, 0.9);
    net::LeakyReLU activation;

    tensor embed_out = embedding.forward(input_tensor);
    tensor activ_out = activation.forward(embed_out);

    tensor targets(input_tensor.batch(), input_tensor.channel(), maxSeqLen, embedding_dim);
    tensor loss = mse.forward(activ_out, targets);
    tensor grad_loss = mse.backward(activ_out, targets);

    tensor grad_activ = activation.backward(grad_loss);
    tensor embed_grad_out = embedding.backward(grad_activ);

    auto params = embedding.parameters();
    auto grads = embedding.gradients();
    for (size_t i = 0; i < params.size(); ++i) {
        optimizer.add_param(params[i], grads[i]);
    }
    optimizer.step();

    log("Embedding parameters (Before optimizer): ");
    embed_grad_out.print();

    std::cout << std::endl;

    log("Loss: ");
    loss.print();

    return 0;
}

/*
----- output -----
[LOG]: Input tensor:
Tensor shape: [1, 1, 5, 1]
Batch 0:
 Channel 0:
 [ 1 ]
 [ 2 ]
 [ 4 ]
 [ 5 ]
 [ 0 ]

[LOG]: Embedding parameters (Before optimizer):
Empty Tensor
Tensor shape: [0, 0, 0, 0]

[LOG]: Loss:
Tensor shape: [1, 1, 1, 1]
Batch 0:
 Channel 0:
 [ 0.315899 ]
*/
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
    nn::Embedding embedding_layer(embedding_dim, vocab_size);

    auto output = embedding_layer.forward(input_tensor);
    log("Output tensor: ");
    output.print();

    std::cout << std::endl;

    auto grad_output = output;
    auto grad_input = embedding_layer.backward(grad_output);
    log("Grad output tensor: ");
    grad_output.print();

    std::cout << "Backward done, grad_weights updated." << std::endl;

    return 0;
}

/*
*  --- output ---
* [LOG]: Input tensor:
* Tensor shape: [1, 1, 5, 1]
* Batch 0:
* Channel 0:
*  [ 1 ]
*  [ 2 ]
*  [ 4 ]
*  [ 5 ]
*  [ 0 ]

* [LOG]: Output tensor:
* Tensor shape: [1, 1, 5, 16]
* Batch 0:
* Channel 0:
*  [ 0.579188 0.807498 0.71796 0.856809 0.526325 0.272841 0.894104 0.0698302 0.258429 0.0413345 0.139374 0.855779 0.573819 0.612758 0.243042 0.633758 ]
*  [ 0.945054 0.121741 0.41903 0.870325 0.559316 0.773218 0.959745 0.294646 0.653366 0.707816 0.16334 0.0463936 0.640263 0.871455 0.218482 0.630835 ]
*  [ 0.888358 0.487528 0.29816 0.474833 0.962265 0.569732 0.448245 0.869701 0.94941 0.195691 0.686307 0.530196 0.938729 0.443641 0.717141 0.0442309 ]
*  [ 0.716377 0.481316 0.516301 0.0254524 0.0954186 0.235981 0.950911 0.715283 0.916791 0.453572 0.324668 0.762139 0.962477 0.367103 0.935506 0.165289 ]
*  [ 0.644312 0.408271 0.266554 0.569968 0.853785 0.246696 0.419656 0.855203 0.00408989 0.135031 0.781821 0.919151 0.0764128 0.572885 0.422099 0.250859 ]


* [LOG]: Grad output tensor:
* Tensor shape: [1, 1, 5, 16]
* Batch 0:
* Channel 0:
*  [ 0.579188 0.807498 0.71796 0.856809 0.526325 0.272841 0.894104 0.0698302 0.258429 0.0413345 0.139374 0.855779 0.573819 0.612758 0.243042 0.633758 ]
*  [ 0.945054 0.121741 0.41903 0.870325 0.559316 0.773218 0.959745 0.294646 0.653366 0.707816 0.16334 0.0463936 0.640263 0.871455 0.218482 0.630835 ]
*  [ 0.888358 0.487528 0.29816 0.474833 0.962265 0.569732 0.448245 0.869701 0.94941 0.195691 0.686307 0.530196 0.938729 0.443641 0.717141 0.0442309 ]
*  [ 0.716377 0.481316 0.516301 0.0254524 0.0954186 0.235981 0.950911 0.715283 0.916791 0.453572 0.324668 0.762139 0.962477 0.367103 0.935506 0.165289 ]
*  [ 0.644312 0.408271 0.266554 0.569968 0.853785 0.246696 0.419656 0.855203 0.00408989 0.135031 0.781821 0.919151 0.0764128 0.572885 0.422099 0.250859 ]

* Backward done, grad_weights updated.
*/
//
// Created by muham on 19.12.2025.
// Test: Dense + ReLU + MAE + Manual SGD
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {
    nn::Dense dense(4, 3);
    net::ReLU relu;

    tensor input(2, 1, 1, 4);
    tensor target(2, 1, 1, 3);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            input.at(i,0,0,j) = static_cast<float>(i*4 + j + 1);
        }
    }

    target.at(0,0,0,0) = 1.0f;
    target.at(0,0,0,1) = 0.0f;
    target.at(0,0,0,2) = -1.0f;

    target.at(1,0,0,0) = 0.5f;
    target.at(1,0,0,1) = -0.5f;
    target.at(1,0,0,2) = 1.0f;

    log("Initial input:");
    input.print();

    for (int epoch = 0; epoch < 5; ++epoch) {
        float lr = 0.01f;
        net::MeanAbsolute loss_fn;
        log("==== Epoch " + std::to_string(epoch) + " ====");

        tensor dense_out = dense.forward(input);
        tensor relu_out = relu.forward(dense_out);

        log("Dense output:");
        dense_out.print();

        log("ReLU output:");
        relu_out.print();

        tensor loss = loss_fn.forward(relu_out, target);
        log("Loss:");
        loss.print();

        tensor grad_loss = loss_fn.backward(relu_out, target);
        tensor grad_relu = relu.backward(grad_loss);
        tensor grad_input = dense.backward(grad_relu);

        log("Grad input:");
        grad_input.print();

        dense.update_weights(lr); // I will delete this function after activation func classes are ready
    }

    log("Training finished.");
    return 0;
}

/*
---- output ------

[LOG]: Initial input:
Tensor shape: [2, 1, 1, 4]
Batch 0:
 Channel 0:
 [ 1 2 3 4 ]

Batch 1:
 Channel 0:
 [ 5 6 7 8 ]

[LOG]: ==== Epoch 0 ====
[LOG]: Dense output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.906519 1.52062 0.141721 ]

Batch 1:
 Channel 0:
 [ 0.0371739 0 0 ]

[LOG]: ReLU output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.906519 1.52062 0.141721 ]

Batch 1:
 Channel 0:
 [ 0.0371739 0 0 ]

[LOG]: Loss:
Tensor shape: [2, 1, 1, 1]
Batch 0:
 Channel 0:
 [ 0.918608 ]

Batch 1:
 Channel 0:
 [ 0.654275 ]

[LOG]: Grad input:
Tensor shape: [2, 1, 1, 4]
Batch 0:
 Channel 0:
 [ 0.141168 3.82355e-09 0.0545774 0 ]

Batch 1:
 Channel 0:
 [ 0 0 0 0 ]

[LOG]: ==== Epoch 1 ====
[LOG]: Dense output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.899852 1.49396 0.135054 ]

Batch 1:
 Channel 0:
 [ 0.0438406 -0.00333333 -0.00333333 ]

[LOG]: ReLU output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.899852 1.49396 0.135054 ]

Batch 1:
 Channel 0:
 [ 0.0438406 0 0 ]

[LOG]: Loss:
Tensor shape: [2, 1, 1, 1]
Batch 0:
 Channel 0:
 [ 0.909719 ]

Batch 1:
 Channel 0:
 [ 0.652053 ]

[LOG]: Grad input:
Tensor shape: [2, 1, 1, 4]
Batch 0:
 Channel 0:
 [ 0.145612 3.82355e-09 0.0534663 0 ]

Batch 1:
 Channel 0:
 [ 0 0 0 0 ]

[LOG]: ==== Epoch 2 ====
[LOG]: Dense output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.886519 1.44062 0.121721 ]

Batch 1:
 Channel 0:
 [ 0.0571739 -0.01 -0.01 ]

[LOG]: ReLU output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.886519 1.44062 0.121721 ]

Batch 1:
 Channel 0:
 [ 0.0571739 0 0 ]

[LOG]: Loss:
Tensor shape: [2, 1, 1, 1]
Batch 0:
 Channel 0:
 [ 0.891941 ]

Batch 1:
 Channel 0:
 [ 0.647609 ]

[LOG]: Grad input:
Tensor shape: [2, 1, 1, 4]
Batch 0:
 Channel 0:
 [ 0.154501 -3.62703e-09 0.051244 0 ]

Batch 1:
 Channel 0:
 [ 0 0 0 0 ]

[LOG]: ==== Epoch 3 ====
[LOG]: Dense output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.866519 1.36062 0.101721 ]

Batch 1:
 Channel 0:
 [ 0.0771739 -0.02 -0.02 ]

[LOG]: ReLU output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.866519 1.36062 0.101721 ]

Batch 1:
 Channel 0:
 [ 0.0771739 0 0 ]

[LOG]: Loss:
Tensor shape: [2, 1, 1, 1]
Batch 0:
 Channel 0:
 [ 0.865275 ]

Batch 1:
 Channel 0:
 [ 0.640942 ]

[LOG]: Grad input:
Tensor shape: [2, 1, 1, 4]
Batch 0:
 Channel 0:
 [ 0.167834 3.82355e-09 0.0479107 0 ]

Batch 1:
 Channel 0:
 [ 0 0 0 0 ]

[LOG]: ==== Epoch 4 ====
[LOG]: Dense output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.839852 1.25396 0.075054 ]

Batch 1:
 Channel 0:
 [ 0.103841 -0.0333333 -0.0333333 ]

[LOG]: ReLU output:
Tensor shape: [2, 1, 1, 3]
Batch 0:
 Channel 0:
 [ 0.839852 1.25396 0.075054 ]

Batch 1:
 Channel 0:
 [ 0.103841 0 0 ]

[LOG]: Loss:
Tensor shape: [2, 1, 1, 1]
Batch 0:
 Channel 0:
 [ 0.829719 ]

Batch 1:
 Channel 0:
 [ 0.632053 ]

[LOG]: Grad input:
Tensor shape: [2, 1, 1, 4]
Batch 0:
 Channel 0:
 [ 0.185612 -3.62703e-09 0.0434663 0 ]

Batch 1:
 Channel 0:
 [ 0 0 0 0 ]

[LOG]: Training finished.

Process finished with exit code 0

 */

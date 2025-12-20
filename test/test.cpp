//
// Created by muham on 19.12.2025.
// Test: Dense + ReLU + MAE + Manual SGD
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

float compute_accuracy(const tensor &y_pred, const tensor &y_true) {
    int correct = 0;
    const int total = y_pred.batch() * y_pred.channel() * y_pred.height();

    for (int i = 0; i < y_pred.batch(); ++i) {
        for (int j = 0; j < y_pred.channel(); ++j) {
            for (int k = 0; k < y_pred.height(); ++k) {
                int pred_label = 0;
                float max_val = y_pred.at(i,j,k,0);
                for (int l = 1; l < y_pred.width(); ++l) {
                    if (y_pred.at(i,j,k,l) > max_val) {
                        max_val = y_pred.at(i,j,k,l);
                        pred_label = l;
                    }
                }

                int true_label = 0;
                float max_true = y_true.at(i,j,k,0);
                for (int l = 1; l < y_true.width(); ++l) {
                    if (y_true.at(i,j,k,l) > max_true) {
                        max_true = y_true.at(i,j,k,l);
                        true_label = l;
                    }
                }

                if (pred_label == true_label) correct++;
            }
        }
    }

    return static_cast<float>(correct) / static_cast<float>(total);
}


int main() {
    constexpr int input_size = 4;
    constexpr int hidden_size = 5;
    constexpr int output_size = 3;
    constexpr int batch_size = 2;

    nn::Dense dense1(input_size, hidden_size);
    nn::Dense dense2(hidden_size, output_size);
    net::ReLU relu;

    tensor x(batch_size,1,1,input_size);
    tensor y_true(batch_size,1,1,output_size);

    x.uniform_rand();
    y_true.uniform_rand();

    net::Adam optimizer(0.01);

    for (nn::Dense* layers[] = { &dense1, &dense2 }; nn::Dense* layer : layers) {
        auto params = layer->parameters();
        auto grads  = layer->gradients();
        for (size_t i = 0; i < params.size(); ++i) {
            optimizer.add_param(params[i], grads[i]);
        }
    }

    for (int step = 0; step < 20; ++step) {
        net::MeanSquared loss;
        tensor h1 = dense1.forward(x);
        tensor h2 = relu.forward(h1);
        tensor y_pred = dense2.forward(h2);

        tensor l = loss.forward(y_pred, y_true);
        float acc = compute_accuracy(y_pred, y_true);
        std::cout << "Step " << step + 1 << ", Loss: " << l.at(0,0,0,0) << ", Accuracy: " << acc << std::endl;

        tensor grad_loss = loss.backward(y_pred, y_true);
        tensor grad_h2 = dense2.backward(grad_loss);
        tensor grad_h1 = relu.backward(grad_h2);
        dense1.backward(grad_h1);

        optimizer.step();
        optimizer.zero_grad();
    }

    std::cout << "Mini training loop completed." << std::endl;
    return 0;
}

/*  ---- output ----
*   Step 1, Loss: 0.135245, Accuracy: 0.5
*   Step 2, Loss: 0.126451, Accuracy: 1
*   Step 3, Loss: 0.11764, Accuracy: 1
*   Step 4, Loss: 0.108938, Accuracy: 1
*   Step 5, Loss: 0.100375, Accuracy: 1
*   Step 6, Loss: 0.0919815, Accuracy: 1
*   Step 7, Loss: 0.0837848, Accuracy: 1
*   Step 8, Loss: 0.0758132, Accuracy: 1
*   Step 9, Loss: 0.0680953, Accuracy: 1
*   Step 10, Loss: 0.0606599, Accuracy: 1
*   Step 11, Loss: 0.0535361, Accuracy: 1
*   Step 12, Loss: 0.0467537, Accuracy: 1
*   Step 13, Loss: 0.0411723, Accuracy: 1
*   Step 14, Loss: 0.0361579, Accuracy: 1
*   Step 15, Loss: 0.0313829, Accuracy: 1
*   Step 16, Loss: 0.0268745, Accuracy: 1
*   Step 17, Loss: 0.0226612, Accuracy: 1
*   Step 18, Loss: 0.0187718, Accuracy: 1
*   Step 19, Loss: 0.0152357, Accuracy: 1
*   Step 20, Loss: 0.012082, Accuracy: 1
*   Mini training loop completed.
*/
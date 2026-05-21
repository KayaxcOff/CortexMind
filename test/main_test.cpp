//
// Created by muham on 7.05.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iomanip>
#include <iostream>

using namespace cortex;

int main() {
    std::cout << "=== XOR Training ===" << std::endl;

    const std::vector x_data = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };
    const std::vector y_data = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    tensor X({4, 2}, x_data.data(), host);
    tensor Y({4, 1}, y_data.data(), host);

    nn::Dense hidden(2, 4, host);
    nn::ReLU  relu;
    nn::Dense output_layer(4, 1, host);

    loss::MeanSquared mse;
    opt::StochasticGradient sgd(0.0000000000000001f);

    auto params = hidden.getParameters();
    auto out_params = output_layer.getParameters();
    params.insert(params.end(), out_params.begin(), out_params.end());
    sgd.SetParams(params);

    std::cout << "=== Weight Initialization ===" << std::endl;

    auto h_params = hidden.getParameters();
    auto o_params = output_layer.getParameters();

    std::cout << "Hidden weight max: " << h_params[0].get().max()
              << " | min: " << h_params[0].get().min() << std::endl;
    std::cout << "Hidden bias max: " << h_params[1].get().max()
              << " | min: " << h_params[1].get().min() << std::endl;

    std::cout << "Output weight max: " << o_params[0].get().max()
              << " | min: " << o_params[0].get().min() << std::endl;
    std::cout << "Output bias max: " << o_params[1].get().max()
              << " | min: " << o_params[1].get().min() << std::endl;

    constexpr int epochs = 30;
    for (int epoch = 0; epoch < epochs; ++epoch) {

        // Forward
        tensor h  = hidden.forward(X);
        std::cout << "h max: " << h.max() << " | h min: " << h.min() << std::endl;

        tensor h2 = relu.forward(h);
        std::cout << "h2 (after ReLU) max: " << h2.max() << std::endl;

        tensor out = output_layer.forward(h2);
        std::cout << "out max: " << out.max() << " | out min: " << out.min() << std::endl;

        tensor l = mse.forward(out, Y);
        std::cout << "Loss: " << l.get()[0] << std::endl;

        std::cout << "h stats - max: " << h.max() << " min: " << h.min()
          << " mean: " << h.mean() << " std: " << h.stdv() << std::endl;

        std::cout << "\n=== Before Backward ===" << std::endl;
        for (size_t i = 0; i < sgd.parameters().size(); ++i) {
            std::cout << "Param " << i << " grad before: "
                      << sgd.parameters()[i].get().grad().max() << std::endl;
        }
        //sgd.zero_grad();
        l.backward();

        std::cout << "\n=== After Backward ===" << std::endl;
        for (size_t i = 0; i < sgd.parameters().size(); ++i) {
            std::cout << "Param " << i << " grad after: "
                      << sgd.parameters()[i].get().grad().max() << std::endl;
        }

        std::cout << "out.grad max: " << out.grad().max()
          << " | mean: " << out.grad().mean() << std::endl;

        auto h_grads = hidden.getGradients();
        std::cout << "hidden.weight.grad max: " << h_grads[0].get().max() << std::endl;

        auto hidden_grads = hidden.getGradients();
        std::cout << "hidden.weight.grad max: " << hidden_grads[0].get().max() << std::endl;
        std::cout << "hidden.bias.grad max: " << hidden_grads[1].get().max() << std::endl;

        auto out_grads = output_layer.getGradients();
        std::cout << "output.weight.grad max: " << out_grads[0].get().max() << std::endl;
        std::cout << "output.bias.grad max: " << out_grads[1].get().max() << std::endl;

        std::cout << "---" << std::endl;

        sgd.update();
        sgd.zero_grad();
    }

    std::cout << "\n=== Predictions ===" << std::endl;
    tensor h   = hidden.forward(X);
    tensor h2  = relu.forward(h);
    tensor pred = output_layer.forward(h2);

    const std::vector expected = {0.0f, 1.0f, 1.0f, 0.0f};
    for (int i = 0; i < 4; ++i) {
        std::cout << "Input: [" << x_data[i*2] << ", " << x_data[i*2+1] << "]"
                  << " | Pred: " << std::fixed << std::setprecision(4)
                  << pred.get()[i]
                  << " | Expected: " << expected[i] << std::endl;
    }

    return 0;
}
/*
C:\software\Cpp\projects\CortexMind\cmake-build-debug-visual-studio\CXM_MAIN_TEST.exe
=== XOR Training ===
=== Weight Initialization ===
Hidden weight max: 0.942912 | min: -0.307687
Hidden bias max: 0 | min: 0
Output weight max: 0.430643 | min: -0.784449
Output bias max: 0 | min: 0
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: 0
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 6.13014e+08 | mean: 6.14034e+07
hidden.weight.grad max: 0
hidden.weight.grad max: 0
hidden.bias.grad max: 1.05772e+08
output.weight.grad max: 7.39854e+07
output.bias.grad max: 2.45614e+08
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 0 | mean: -4.50857e-15
hidden.weight.grad max: 0
hidden.weight.grad max: 0
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 0
output.bias.grad max: -1.80343e-14
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 0
hidden.weight.grad max: 0
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 0
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---
h max: 1.75409 | h min: -0.307687
h2 (after ReLU) max: 1.75409
out max: 0.891294 | out min: -1.27703e-14
Loss: 0.378895
h stats - max: 1.75409 min: -0.307687 mean: 0.288662 std: 0.564764
out.grad max: 5.60519e-45 | mean: -1.59629e-15
hidden.weight.grad max: 2.8026e-45
hidden.weight.grad max: 2.8026e-45
hidden.bias.grad max: 2.32862e-15
output.weight.grad max: 4.2039e-45
output.bias.grad max: -6.38516e-15
---

=== Predictions ===
Input: [0, 0] | Pred: -0.0000 | Expected: 0.0000
Input: [0.0000, 1.0000] | Pred: 0.5343 | Expected: 1.0000
Input: [1.0000, 0.0000] | Pred: 0.2898 | Expected: 1.0000
Input: [1.0000, 1.0000] | Pred: 0.8913 | Expected: 0.0000

Process finished with exit code 0
*/
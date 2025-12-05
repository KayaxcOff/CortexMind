//
// Created by muham on 30.11.2025.
//

#include "../include/CortexMind/cortexmind.hpp"
#include <vector>

using namespace cortex;

int main() {
    model::Model net;

    size_t num_samples = 10;
    size_t out_feats = 1;
    size_t in_feats = 5;

    std::vector<tensor> input = {};
    std::vector<tensor> output = {};

    for (size_t i = 0; i < num_samples; i++) {
        tensor x(1, in_feats, false);
        for(size_t j=0; j < in_feats; ++j) {
            x.get_data()[j] = static_cast<double>(i + j) / 10.0;
        }
        input.push_back(std::move(x));

        tensor y(1, out_feats, false);
        y.get_data()[0] = 0.5;
        output.push_back(std::move(y));
    }

    tensor test(1, in_feats, false);
    for(size_t j=0; j < in_feats; ++j) {
        test.get_data()[j] = static_cast<double>(j) / 10.0;
    }

    net.add<nn::Dense>(in_feats, 10);
    net.add<nn::Flatten>();
    net.add<nn::Dense>(10, out_feats);

    net.compile<loss::MeanAbsolute, optim::Adam, act::ReLU>();

    net.train(input, output, 5, 5);

    const auto pred = net.predict(test);
    pred.print();

    return 0;
}
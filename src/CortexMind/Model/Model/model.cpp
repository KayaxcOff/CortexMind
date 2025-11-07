//
// Created by muham on 4.11.2025.
//

#include "CortexMind/Model/Model/model.hpp"

#include <iostream>
#include <ranges>

using namespace cortex::model;
using namespace cortex;

void Model::fit(const std::vector<math::MindVector> &X, const std::vector<math::MindVector> &Y, const size_t epochs) {
    if (X.size() != Y.size()) {
        throw std::runtime_error("Model::fit -> X and Y size mismatch");
    }

    for (size_t i = 0; i < epochs; ++i) {
        double epoch_loss = 0.0;

        for (size_t j = 0; j < X.size(); ++j) {
            math::MindVector output = X[j];

            for (const auto& layer : layers_) {
                output = layer->forward(output);
            }

            if (output.size() != Y[j].size()) {
                throw std::runtime_error("Model::fit -> output/target size mismatch at sample " + std::to_string(j));
            }

            epoch_loss += loss_fn_->forward(output, Y[j]);
            math::MindVector loss_grad = loss_fn_->backward(output, Y[j]);

            for (const auto & layer : std::ranges::reverse_view(layers_)) {
                loss_grad = layer->backward(loss_grad);
            }

            optim_fn_->step();
        }
        std::cout << "Epoch: " << i + 1 << "/" << epochs << " - Loss: " << epoch_loss / static_cast<double>(X.size()) << std::endl;
    }
}

math::MindVector Model::predict(const math::MindVector &X) const {
    math::MindVector output = X;
    for (const auto& layer : layers_) {
        output = layer->forward(output);
    }
    return output;
}
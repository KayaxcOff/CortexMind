//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/Model/Model/model.hpp"
#include <CortexMind/framework/Log/log.hpp>
#include <numeric>
#include <random>

using namespace cortex::model;
using namespace cortex;

void Model::summary() const {
    for (const auto& item : this->layers_) {
        log("\n" + item->config());
    }
}

void Model::train(std::vector<tensor> &_x, std::vector<tensor> &_y, int epochs, int batchSize) {
    if (!this->loss_fn_ || !this->optim_fn_ || !this->activ_fn_) {
        log("Model is not compiled");
        throw std::logic_error("Model is not compiled");
    }

    if (_x.size() != _y.size()) {
        log("X and Y sizes don't match");
        throw std::logic_error("X and Y sizes don't match");
    }

    std::vector<optim::TensorParams> tensor_params;
    for (const auto& item : this->layers_) {
        auto weights = item->getParameters();
        auto grads   = item->getGradients();

        for (size_t i = 0; i < weights.size(); ++i)
            tensor_params.push_back({weights[i], grads[i]});
    }
    this->optim_fn_->register_parameters(tensor_params);

    size_t num_samples = _x.size();
    size_t num_batches = (num_samples + batchSize - 1) / batchSize;

    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    for (int i = 0; i <= epochs; ++i) {
        std::ranges::shuffle(indices, std::mt19937{std::random_device{}()});

        double epoch_loss = 0.0;

        for (size_t j = 0; j < num_batches; ++j) {
            this->optim_fn_->zero_grad();

            size_t startIdx = j * batchSize;

            const tensor& inputBatch  = _x[indices[startIdx]];
            const tensor& targetBatch = _y[indices[startIdx]];

            tensor pred = inputBatch;
            for (auto& item : this->layers_) {
                pred = item->forward(pred);
            }

            tensor loss_t = this->loss_fn_->forward(pred, targetBatch);
            double loss_value = loss_t.get_data()[0];
            epoch_loss += loss_value;

            tensor grad = this->loss_fn_->backward(pred, targetBatch);

            grad = this->activ_fn_->backward(grad);

            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                grad = (*it)->backward(grad);
            }

            this->optim_fn_->step();
        }
        std::cout << "Epoch " << i + 1 << "/" << epochs << " - Loss: " << (epoch_loss / static_cast<double>(num_batches)) << std::endl;
    }
}

void Model::forget() const {
    if (this->optim_fn_) {
        return;
    }

    for (const auto& item : this->layers_) {
        std::vector<tensor*> weights = item->getParameters();
        std::vector<tensor*> grads = item->getGradients();

        for (size_t i = 0; i < weights.size(); ++i) {
            tensor* weight = weights[i];

            if (const tensor* grad = grads[i]; !weight || !grad) {
                continue;
            }

            const size_t size = weight->get_data().size();
            for (size_t j = 0; j < size; ++j) {
                constexpr double learning_rate = 0.01;
                weight->get_data()[j] += learning_rate * weight->get_data()[j];
            }
        }
    }
}

tensor Model::predict(const tensor& input) const {
    if (!this->layers_.empty()) {
        log("Only one layer allowed");
        throw std::logic_error("Only one layer allowed");
    }

    tensor output = input;
    for (auto& item : this->layers_) {
        output = item->forward(output);
    }

    if (this->activ_fn_) {
        output = this->activ_fn_->forward(output);
    }

    return output;
}
//
// Created by muham on 24.05.2026.
//

#include "CortexMind/net/Model/model.hpp"
#include <algorithm>
#include <iostream>

using namespace cortex::net;
using namespace cortex;

Model::Model() : flag(false) {}

Model::~Model() = default;

void Model::fit(const tensor &Xx, const tensor &Xy, int32 epochs, int32 batch) const {
    CXM_ASSERT(!this->flag, "Model isn't compile");
    CXM_ASSERT(!this->trainable(), "Model can't trainable");
}

void Model::summary() const {
    for (const auto& item : this->layers_) {
        std::cout << item->name() << std::endl;
    }
}

bool Model::trainable() const {
    return std::ranges::any_of(
        this->layers_,
        [](const auto& item) {
            return item->flag();
        }
    );
}

void Model::toTrain() const {
    for (const auto& item : this->layers_) {
        item->TrainMode();
    }
}

void Model::toEval() const {
    for (const auto& item : this->layers_) {
        item->EvalMode();
    }
}

tensor Model::predict(const tensor &x) const {
    tensor output = x;
    for (const auto& item : this->layers_) {
        output = item->forward(output);
    }
    return output;
}

std::vector<_fw::ref<tensor>> Model::parameters() const {
    std::vector<_fw::ref<tensor>> output;
    for (const auto & item : this->layers_) {
        for (size_t i = 0; i < item->getParameters().size(); ++i) {
            output.push_back(item->getParameters()[i]);
        }
    }
    return output;
}

std::vector<_fw::ref<tensor>> Model::gradients() const {
    std::vector<_fw::ref<tensor>> output;
    for (const auto & item : this->layers_) {
        for (size_t i = 0; i < item->getGradients().size(); ++i) {
            output.push_back(item->getGradients()[i]);
        }
    }
    return output;
}
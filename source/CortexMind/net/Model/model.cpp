//
// Created by muham on 24.05.2026.
//

#include "CortexMind/net/Model/model.hpp"
#include <algorithm>
#include <iomanip>
#include <ios>
#include <iostream>

using namespace cortex::net;
using namespace cortex;

Model::Model(std::string name) : m_flag(false), m_name(std::move(name)) {}

Model::~Model() = default;

void Model::fit(const tensor &Xx, const tensor &Xy, const int32 epochs, int32 epochIdx) const {
    CXM_ASSERT(!this->m_flag, "Model isn't compiled");

    this->optim_fn_->SetParams(this->parameters());

    for (int32 epoch = 0; epoch < epochs; ++epoch) {

        tensor pred = this->predict(Xx);

        tensor loss = this->loss_fn_->forward(pred, Xy);
        const float32 epoch_loss = loss.get()[0];

        this->optim_fn_->zero_grad();

        loss.backward();

        this->optim_fn_->update();

        if (epoch % epochIdx == 0) {
            std::cout
            << "Epoch "
            << std::setw(5)
            << epoch
            << " | Loss: "
            << std::fixed
            << std::setprecision(6)
            << epoch_loss
            << "%"
            << std::endl;
        }
    }
}

tensor Model::predict(const tensor &x) const {
    tensor output = x;
    for (const auto& item : this->layers_) {
        output = item->forward(output);
    }
    return output;
}

void Model::summary() const {

    constexpr int W1 = 30;
    constexpr int W2 = 15;

    std::cout << "\n==================================================\n";
    std::cout << "Model: " <<(this->m_name.empty() ? this->m_name : "") << '\n';
    std::cout << "==================================================\n";

    std::cout << std::left << std::setw(W1) << "Layer" << std::setw(W2) << "Trainable" << '\n';

    std::cout << "--------------------------------------------------\n";

    for (const auto& item : this->layers_) {
        std::cout << std::left << std::setw(W1) << item->name() << std::setw(W2) << (item->flag() ? "Yes" : "No") << '\n';
    }

    std::cout << "==================================================\n";

    std::cout << "Is compiled   : " << (this->m_flag ? "Yes" : "No") << '\n';

    std::cout << "Loss Function : " << (this->loss_fn_ ? this->loss_fn_->name() : "None") << '\n';

    std::cout << "Optimizer     : "<< (this->optim_fn_ ? this->optim_fn_->name() : "None") << '\n';

    std::cout << "Total Params  : " << this->compute_element() << '\n';

    std::cout << "==================================================\n";
}

void Model::train() const {
    for (const auto& item : this->layers_) {
        item->TrainMode();
    }
}

void Model::eval() const {
    for (const auto& item : this->layers_) {
        item->EvalMode();
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

size_t Model::compute_element() const {
    size_t output = 0;

    for (const auto& item : this->layers_) {
        for (size_t i = 0; i < item->getParameters().size(); ++i) {
            output += item->getParameters()[i].get().len();
        }
    }
    return output;
}

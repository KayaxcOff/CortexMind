//
// Created by muham on 24.05.2026.
//

#include "CortexMind/net/Model/model.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>

using namespace cortex::net;
using namespace cortex;


Model::Model(std::string name) : m_flag(false), m_name(std::move(name)) {}

Model::~Model() = default;

void Model::fit(const tensor &Xx, const tensor &Xy, int32 epochs, int32 batch) const {
    CXM_ASSERT(!this->m_flag, "Model isn't compiled");

    this->optimization_fn_->SetParams(this->parameters());

    const auto N        = static_cast<int64>(Xx.shape()[0]);
    const int64 n_batch  = (N + batch - 1) / batch;

    for (int32 epoch = 0; epoch < epochs; ++epoch) {
        float32 epoch_loss = 0.0f;

        for (int64 b = 0; b < n_batch; ++b) {
            const int64 start = b * batch;
            const int64 end   = std::min(start + batch, N);

            const tensor Xb = Xx.slice(0, start, end);
            const tensor Yb = Xy.slice(0, start, end);

            const tensor pred = this->predict(Xb);

            const tensor loss = this->loss_fn_->forward(pred, Yb);
            epoch_loss += loss.get()[0];

            this->optimization_fn_->zero_grad();
            loss.backward();
            this->optimization_fn_->update();
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << std::setw(5) << epoch
                      << " | Loss: " << std::fixed
                      << std::setprecision(6)
                      << epoch_loss / static_cast<float32>(n_batch)
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
    using std::cout;
    using std::left;
    using std::setw;

    constexpr int W1 = 30;
    constexpr int W2 = 15;

    cout << "\n==================================================\n";
    cout << "Model: " <<(this->m_name.empty() ? this->m_name : "") << '\n';
    cout << "==================================================\n";

    cout << left
         << setw(W1) << "Layer"
         << setw(W2) << "Trainable"
         << '\n';

    cout << "--------------------------------------------------\n";

    for (const auto& layer : this->layers_) {
        cout << left
             << setw(W1) << layer->name()
             << setw(W2) << (layer->flag() ? "Yes" : "No")
             << '\n';
    }

    cout << "==================================================\n";

    cout << "Loss Function : "
         << (this->loss_fn_ ? this->loss_fn_->name() : "None")
         << '\n';

    cout << "Optimizer     : "
         << (this->optimization_fn_ ? this->optimization_fn_->name() : "None")
         << '\n';

    cout << "Total Params  : "
         << this->compute_element()
         << '\n';

    cout << "==================================================\n";
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

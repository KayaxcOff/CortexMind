//
// Created by muham on 1.03.2026.
//

#include "CortexMind/net/Model/model.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

Model::Model() {
    this->compile_flag = false;
    this->callback_flag = false;
}

Model::~Model() = default;

void Model::fit(tensor &_x, tensor &_y, const int64 _epochs, const int64 _batch) const {
    CXM_ASSERT(this->compile_flag, "cortex::net::Model::fit()", "Model not compiled");
    CXM_ASSERT(!this->layers_.empty(), "cortex::net::Model::fit()", "There is no layer");
    CXM_ASSERT(_x.shape()[0] == _y.shape()[0], "cortex::net::Model::fit()", "X and Y batch size mismatch.");

    this->optim_fn_->setParams(this->collect_params(), this->collect_grads());

    const int64 n = _x.shape()[0];
    const int64 batch_size = (_batch == -1) ? n : _batch;

    if (this->callback_flag) {
        for (const auto& item : this->callbacks_) {
            item->on_train_begin();
        }
    }

    for (int64 i = 0; i < _epochs; ++i) {

        if (this->callback_flag) {
            for (const auto& item : this->callbacks_) {
                item->on_epoch_begin(i + 1);
            }
        }

        float32 epoch_loss = 0.0f;
        int64 n_batches = 0;
        for (int64 j = 0; j < n; j += batch_size) {
            int64 end = std::min(j + batch_size, n);

            if (this->callback_flag) {
                for (const auto& item : this->callbacks_) {
                    item->on_batch_begin(j / batch_size);
                }
            }

            tensor x_batch = _x.slice(j, end);
            tensor y_batch = _y.slice(j, end);

            this->optim_fn_->zero_grad();

            tensor pred = this->predict(x_batch);
            tensor loss = this->loss_fn_->forward(pred, y_batch);
            loss.backward();
            this->optim_fn_->update();

            epoch_loss += loss.at(0);
            ++n_batches;

            if (this->callback_flag) {
                for (const auto& item : this->callbacks_) {
                    item->on_batch_end(j / batch_size, loss.at(0));
                }
            }
        }
        for (const auto& item : this->layers_) item->set_train(false);
        tensor pred_eval = this->predict(_x);
        const float32 avg_loss = epoch_loss / static_cast<float32>(n_batches);
        const float32 acc = accuracy(pred_eval, _y);
        std::cout << "["  << std::setw(static_cast<int32>(std::to_string(_epochs).size()))
                  << (i + 1) << "/" << _epochs << "]"
                  << " Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Acc: " << std::fixed << std::setprecision(2) << acc << "%"
                  << "\n";
        if (this->callback_flag) {
            for (const auto& item : this->callbacks_) item->on_epoch_end(i + 1, avg_loss, acc);

            bool stop = false;
            string name;
            for (const auto& item : this->callbacks_)
                if (item->shouldStop()) {
                    stop = true;
                    name = item->config();
                    break;
                }

            if (stop) {
                std::cout << "[" << name << "] Training stopped at epoch " << (i + 1) << "\n";
                break;
            }
        }
    }
    if (this->callback_flag) {
        for (const auto& item : this->callbacks_) {
            item->on_train_end();
        }
    }
    this->eval_mode();
}

void Model::train_mode() const {
    for (const auto& item : this->layers_) item.get()->set_train(true);
}

void Model::eval_mode() const {
    for (const auto& item : this->layers_) item.get()->set_train(false);
}

tensor Model::predict(tensor &test) const {
    CXM_ASSERT(!this->layers_.empty(), "cortex::net::Model::predict()", "No layers.");

    tensor* output = &test;
    tensor temp;

    for (const auto& item : this->layers_) {
        temp = item->forward(*output);
        output = &temp;
    }

    return temp;
}

void Model::summary() const {
    std::cout << "\n<====================== CXM - Model =====================>\n\n";

    std::cout << "Layers\n";
    std::cout << "-----------------------------------------------------------------\n";

    std::cout << std::left
              << std::setw(4)  << "#"
              << " | "
              << std::setw(18) << "Type"
              << " | "
              << std::setw(20) << "Config"
              << " | "
              << "Params\n";

    std::cout << "-----------------------------------------------------------------\n";

    size_t total_params = 0;
    size_t i = 0;

    for (const auto& item : this->layers_) {

        string cfg = item->config();
        string type = cfg.substr(0, cfg.find('('));
        if (type.empty()) type = cfg;

        size_t param_count = 0;

        for (const auto& p : item->parameters())
            param_count += p.get().numel();

        total_params += param_count;

        std::cout << std::left
                  << std::setw(4)  << i++
                  << " | "
                  << std::setw(18) << type
                  << " | "
                  << std::setw(20) << cfg
                  << " | "
                  << std::right << std::setw(6) << param_count
                  << std::left
                  << "\n";
    }

    std::cout << "-----------------------------------------------------------------\n\n";

    std::cout << "Total params: " << total_params << "\n\n";

    std::cout << std::left << std::setw(10) << "Loss"
              << " : " << this->loss_fn_->config() << "\n";

    std::cout << std::left << std::setw(10) << "Optimizer"
              << " : " << this->optim_fn_->config() << "\n";

    std::cout << std::left << std::setw(10) << "Callbacks" << " : ";
    for (const auto& item : this->callbacks_) {
        std::cout << std::left << std::setw(12) << item->config() << std::endl;
    }

    std::cout << "\n<========================================================>\n";
}

float32 Model::accuracy(const tensor &predicted, const tensor &target) {
    CXM_ASSERT(predicted.shape()[0] == target.shape()[0], "cortex::net::Model::accuracy()", "Batch size mismatch.");

    const int64 n = predicted.shape()[0];
    int64 correct = 0;

    for (int64 i = 0; i < n; ++i) {
        if (predicted.shape()[1] == 1) {
            const float32 pred  = std::round(predicted.at(i, 0));
            const float32 truth = std::round(target.at(i, 0));
            if (pred == truth) ++correct;
        } else {
            int64 pred_class = 0, true_class = 0;
            for (int64 j = 1; j < predicted.shape()[1]; ++j) {
                if (predicted.at(i, j) > predicted.at(i, pred_class)) pred_class = j;
                if (target.at(i, j)    > target.at(i, true_class))    true_class = j;
            }
            if (pred_class == true_class) ++correct;
        }
    }
    return static_cast<float32>(correct) / static_cast<float32>(n) * 100.0f;
}

std::pair<float32, float32> Model::evaluate(tensor &_x, const tensor &_y) const {
    const tensor preds = this->predict(_x);
    tensor loss = this->loss_fn_->forward(preds, _y);
    const float32 acc = accuracy(preds, _y);
    return {loss.at(0), acc};
}

std::vector<ref<tensor>> Model::collect_params() const {
    std::vector<ref<tensor>> all;
    for (const auto& item : this->layers_) {
        auto p = item->parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<ref<tensor>> Model::collect_grads() const {
    std::vector<ref<tensor>> all;
    for (const auto& item : this->layers_) {
        auto p = item->gradients();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

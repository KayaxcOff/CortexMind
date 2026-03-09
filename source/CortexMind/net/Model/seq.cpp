//
// Created by muham on 9.03.2026.
//

#include "CortexMind/net/Model/seq.hpp"
#include <iomanip>

using namespace cortex::net;
using namespace cortex::_fw;
using namespace cortex;

void Sequential::fit(tensor &_x, tensor &_y, int64 _epochs, int64 _batch, const std::vector<std::unique_ptr<Callback> >& callbacks) const {
    CXM_ASSERT(this->compile_flag, "cortex::net::Sequential::fit()", "Model not compiled");
    CXM_ASSERT(!this->layers_.empty(), "cortex::net::Sequential::fit()", "There is no layer");
    CXM_ASSERT(_x.shape()[0] == _y.shape()[0], "cortex::net::Sequential::fit()", "X and Y batch size mismatch.");

    this->optim_fn_->setParams(this->collect_params(), this->collect_grads());

    const int64 n          = _x.shape()[0];
    const int64 batch_size = (_batch == -1) ? n : _batch;
    const bool  has_cb     = !callbacks.empty();

    if (has_cb)
        for (const auto& item : callbacks) item->on_train_begin();

    for (int64 i = 0; i < _epochs; ++i) {

        if (has_cb)
            for (const auto& cb : callbacks) cb->on_epoch_begin(i + 1);

        float32 epoch_loss = 0.0f;
        int64   n_batches  = 0;

        for (int64 j = 0; j < n; j += batch_size) {
            const int64 end = std::min(j + batch_size, n);

            if (has_cb)
                for (const auto& item : callbacks) item->on_batch_begin(j / batch_size);

            tensor x_batch = _x.slice(j, end);
            tensor y_batch = _y.slice(j, end);

            this->optim_fn_->zero_grad();

            tensor pred = this->predict(x_batch);
            tensor loss = this->loss_fn_->forward(pred, y_batch);
            loss.backward();
            this->optim_fn_->update();

            epoch_loss += loss.at(0);
            ++n_batches;

            if (has_cb)
                for (const auto& item : callbacks) item->on_batch_end(j / batch_size, loss.at(0));
        }

        this->eval_mode();
        tensor pred_eval       = this->predict(_x);
        const float32 avg_loss = epoch_loss / static_cast<float32>(n_batches);
        const float32 acc      = accuracy(pred_eval, _y);
        this->train_mode();

        std::cout << "[" << std::setw(static_cast<int>(std::to_string(epochs).size()))
                  << (i + 1) << "/" << epochs << "]"
                  << " Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Acc: " << std::fixed << std::setprecision(2) << acc << "%"
                  << "\n";

        if (has_cb) {
            for (const auto& cb : callbacks) cb->on_epoch_end(i + 1, avg_loss, acc);

            std::string stop_name;
            for (const auto& item : callbacks)
                if (item->shouldStop()) { stop_name = item->config(); break; }

            if (!stop_name.empty()) {
                std::cout << "[" << stop_name << "] Training stopped at epoch " << (i + 1) << "\n";
                break;
            }
        }
    }

    if (has_cb)
        for (const auto& cb : callbacks) cb->on_train_end();

    this->eval_mode();
}

void Sequential::summary() const {
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

    std::cout << "\n<========================================================>\n";
}

void Sequential::train_mode() const {
    for (const auto& item : this->layers_) item.get()->set_train(true);
}

void Sequential::eval_mode() const {
    for (const auto& item : this->layers_) item.get()->set_train(false);
}

tensor Sequential::predict(tensor &input) const {
    CXM_ASSERT(!this->layers_.empty(), "cortex::net::Model::predict()", "No layers.");

    tensor* output = &input;
    tensor temp;

    for (const auto& item : this->layers_) {
        temp = item->forward(*output);
        output = &temp;
    }

    return temp;
}

std::vector<ref<tensor>> Sequential::collect_params() const {
    std::vector<ref<tensor>> all;
    for (const auto& item : this->layers_) {
        auto p = item->parameters();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

std::vector<ref<tensor>> Sequential::collect_grads() const {
    std::vector<ref<tensor>> all;
    for (const auto& item : this->layers_) {
        auto p = item->gradients();
        all.insert(all.end(), p.begin(), p.end());
    }
    return all;
}

float32 Sequential::accuracy(const tensor &predicted, const tensor &target) {
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

//
// Created by muham on 24.05.2026.
//

#include "CortexMind/net/Model/model.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>

using namespace cortex::net;
using namespace cortex;

Model::Model(std::string name) : m_flag(false), m_name(std::move(name)) {}

Model::~Model() = default;

void Model::fit(const tensor &Xx, const tensor &Xy, const int32 epochs, const int32 epochIdx) const {
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
    std::cout << "Model: " <<(this->m_name.empty() ? this->m_name : "Model") << '\n';
    std::cout << "==================================================\n";

    std::cout << std::left << std::setw(W1) << "Layer" << std::setw(W2) << "Mode" << '\n';

    std::cout << "--------------------------------------------------\n";

    for (const auto& item : this->layers_) {
        std::cout << std::left << std::setw(W1) << item->name() << std::setw(W2) << (item->flag() ? "Train" : "Eval") << '\n';
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

void Model::save(const std::string &path) {
    std::filesystem::create_directories(path);

    nlohmann::ordered_json json;
    json["model_name"] = this->m_name;
    json["loss_function"] = this->loss_fn_->name();
    json["optimizer"] = this->optim_fn_->name();
    json["layers"] = nlohmann::json::array();

    std::ofstream bin(path + "/weights.bin", std::ios::binary);

    size_t current_offset = 0;

    for (const auto& item : this->layers_) {
        nlohmann::ordered_json layer_json;
        layer_json["name"] = item->name();
        layer_json["mode"] = item->flag() ? "Train" : "Eval";
        layer_json["params"] = nlohmann::json::array();

        const auto& params = item->getParameters();

        for (size_t i = 0; i < params.size(); ++i) {
            const tensor& t = params[i].get();
            const size_t num_elements = t.len();
            const size_t byte_size   = num_elements * sizeof(float32);

            nlohmann::ordered_json param_json;
            param_json["index"]     = i;
            param_json["offset"]    = current_offset;
            param_json["byte_size"] = byte_size;
            param_json["elements"]  = num_elements;
            layer_json["params"].push_back(param_json);

            bin.write(reinterpret_cast<const char*>(t.get()), static_cast<long long>(byte_size));
            current_offset += byte_size;
        }

        json["layers"].push_back(layer_json);
    }

    bin.close();

    std::ofstream file(path + "/map.json");
    file << json.dump(4);
    file.close();
}

void Model::load(const std::string &path) {
    std::ifstream file(path + "/map.json");
    CXM_ASSERT(!file.is_open(), "map.json can't open");

    nlohmann::ordered_json json;
    file >> json;
    file.close();

    // Temel kontroller
    CXM_ASSERT(json["model_name"] != this->m_name,
        "Model names mismatch");
    CXM_ASSERT(json["layers"].size() != this->layers_.size(),
        "Size of layers mistach");

    std::ifstream bin(path + "/weights.bin", std::ios::binary);
    CXM_ASSERT(!bin.is_open(), "weights.bin can't open");

    for (size_t li = 0; li < this->layers_.size(); ++li) {
        const auto& item       = this->layers_[li];
        const auto& layer_json = json["layers"][li];

        CXM_ASSERT(layer_json["name"] != item->name(),
            "Layer name mismatch: Expected=" + item->name() +
            " In file=" + layer_json["name"].get<std::string>());

        const auto& params = item->getParameters();

        if (params.empty()) {
            continue;
        }

        CXM_ASSERT(layer_json["params"].size() != params.size(),
            "Len of tensor mismatch: " + item->name());

        for (size_t pi = 0; pi < params.size(); ++pi) {
            const auto& param_json = layer_json["params"][pi];
            tensor& t = params[pi].get();

            const size_t expected_elements = param_json["elements"].get<size_t>();
            const size_t offset = param_json["offset"].get<size_t>();
            const size_t byte_size = param_json["byte_size"].get<size_t>();

            CXM_ASSERT(t.len() != expected_elements,
                "Tensor size mismatch: " + item->name() +
                " param[" + std::to_string(pi) + "]");

            bin.seekg(static_cast<std::streamoff>(offset));
            bin.read(reinterpret_cast<char*>(t.get()), static_cast<long long>(byte_size));

            CXM_ASSERT(bin.fail(), "Binary reading error: " + item->name());
        }
    }

    bin.close();
}

bool Model::trainable() const {
    return std::ranges::any_of(
        this->layers_,
        [](const auto& item) {
            return item->flag();
        }
    );
}
/*
std::vector<_fw::ref<tensor>> Model::parameters() const {
    std::vector<_fw::ref<tensor>> output;
    for (const auto & item : this->layers_) {
        for (size_t i = 0; i < item->getParameters().size(); ++i) {
            output.push_back(item->getParameters()[i]);
        }
    }
    return output;
}
*/

std::vector<_fw::ref<tensor>> Model::parameters() const {
    std::vector<_fw::ref<tensor>> output;
    for (const auto& item : this->layers_) {
        for (auto& p : item->getParameters()) {
            output.push_back(p);
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
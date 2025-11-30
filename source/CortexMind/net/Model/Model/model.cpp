//
// Created by muham on 30.11.2025.
//

#include "CortexMind/net/Model/Model/model.hpp"
#include <CortexMind/framework/Log/log.hpp>

using namespace cortex::model;
using namespace cortex;

void Model::summary() const {
    for (const auto& item : this->layers_) {
        log(item->config());
    }
}

tensor Model::predict(const tensor& input) const {
    tensor output = input;
    output.zero();
    for (const auto& layer : this->layers_) {
        output = layer->forward(output);
    }
    return output;
}
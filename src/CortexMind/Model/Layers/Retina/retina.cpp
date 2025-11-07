//
// Created by muham on 3.11.2025.
//

#include "CortexMind/Model/Layers/Retina/retina.hpp"

#include <iostream>
#include <functional>
#include <utility>
#include <stdexcept>
#include <STB/stb_image.h>

using namespace cortex::layer;
using namespace cortex;

Retina::Retina(std::string _path) {
    this->data = nullptr;
    this->path = std::move(_path);
}

Retina::~Retina() {
    if (this->data) stbi_image_free(this->data);
}

math::MindVector Retina::forward(const math::MindVector &input) {
    if (this->data) stbi_image_free(this->data);

    int width, height, channels;

    this->data = stbi_load(this->path.c_str(), &width, &height, &channels, 0);

    if (!this->data) {
        std::cerr << "Cortex Error: " << stbi_failure_reason() << std::endl;
        throw std::runtime_error("Retina::forward -> Failed to load image from path: " + this->path);
    }

    const size_t totalSize = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels);

    std::vector<int> pixels;
    pixels.reserve(totalSize);

    for (size_t i = 0; i < totalSize; i++) {
        pixels.push_back(this->data[i]);
    }

    math::MindVector output;
    output.resize(totalSize);

    for (const size_t i : pixels) {
        output[i] = static_cast<double>(pixels[i]) / 255.0;
    }

    return output;
}

math::MindVector Retina::backward(const math::MindVector &grad_output) {
    return {};
}

void Retina::update(double lr) {}
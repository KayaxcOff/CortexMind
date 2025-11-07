//
// Created by muham on 4.11.2025.
//

#include "CortexMind/Utils/ImageData/image_data.hpp"

#include <STB/stb_image.h>
#include <stdexcept>

using namespace cortex::tools;
using namespace cortex;

MindImage::MindImage() {
    this->path = "";
}

std::vector<math::MindVector> MindImage::transformPixels(const std::string &_path) {

    int width, height, channels;
    unsigned char *data = stbi_load(_path.c_str(), &width, &height, &channels, 0);
    std::vector<math::MindVector> pixels;

    if (data) {pixels.resize(width * height * channels);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                const int index = (i * width + j) * channels;
                const float r = static_cast<float>(data[index]) / 255.0f;
                const float g = static_cast<float>(data[index + 1]) / 255.0f;
                const float b = static_cast<float>(data[index + 2]) / 255.0f;
                pixels.emplace_back(r * g* b);
            }
        }
        stbi_image_free(data);
    } else {
        throw std::runtime_error("Failed to load image: " + _path);
    }

    return pixels;
}
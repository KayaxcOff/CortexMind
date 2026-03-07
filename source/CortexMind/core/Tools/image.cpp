//
// Created by muham on 2.03.2026.
//

#include "CortexMind/core/Tools/image.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <CortexMind/core/Tools/file_system.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <STB/stb_image.h>

using namespace cortex::_fw;
using namespace cortex;

tensor ImageLoader::load(const string &path, const bool gray_scale) {
    int32 width, height, channel;
    const int32 channels = gray_scale ? 1 : 3;

    uint8* data = stbi_load(path.c_str(), &width, &height, &channel, channels);

    CXM_ASSERT(data != nullptr, "cortex::_fw::ImageLoader::load():" ,"Failed to load image.");

    tensor output({channels, height, width});
    for (int32 i = 0; i < channels; ++i) {
        for (int32 j = 0; j < height; ++j) {
            for (int32 k = 0; k < width; ++k) {
                output.at(i, j, k) = static_cast<float32>(data[j * width * channels + k * channels + i]) / 255.0f;
            }
        }
    }
    stbi_image_free(data);
    return output;
}

std::vector<tensor> ImageLoader::load_folder(const string &path, const bool gray_scale) {
    std::vector<tensor> images;
    const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};

    for (const auto& entry : fs::directory_iterator(path)) {
        const std::string ext = entry.path().extension().string();
        if (std::ranges::find(extensions, ext) != extensions.end()) {
            images.push_back(load(entry.path().string(), gray_scale));
        }
    }
    return images;
}

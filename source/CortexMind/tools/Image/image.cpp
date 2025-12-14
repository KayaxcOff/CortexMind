//
// Created by muham on 13.12.2025.
//

#define STB_IMAGE_IMPLEMENTATION

#include "CortexMind/tools/Image/image.hpp"
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <CortexMind/framework/Tools/DataTypes/type.hpp>
#include <CortexMind/framework/Core/AVX/funcs.hpp>
#include <STB/stb_image.h>
#include <iostream>

using namespace cortex::tools;
using namespace cortex::_fw;
using namespace cortex::_fw::avx2;
using namespace cortex;

tensor ImageNetLoader::load(const std::string &path, const bool normalize, const std::optional<std::array<float, 3> > &mean, const std::optional<std::array<float, 3> > &std, const int target_channels) {
    tensor output;

    int width, height, orig_channels;
    uint8* data = stbi_load(path.c_str(), &width, &height, &orig_channels, target_channels);

    if (!data) {
        CXM_ASSERT(true, "Failed to load image: " + path);
    }

    ImageInfo info;
    info.width = width;
    info.height = height;
    info.org_channels = orig_channels;
    info.load_channels = target_channels;
    info.path = path;

    auto spec = TensorSpec::from_image_info(info, target_channels);
    spec.normalize = normalize;
    spec.mean = mean;
    spec.std = std;

    output.allocate(spec.batch, spec.channels, height, width);

    const float norm_factor = normalize ? (1.0f / 255.0f) : 1.0f;

    const uint8* src = data;
    const size_t stride = width * target_channels;

    for (int i = 0; i < target_channels; ++i) {
        for (int j = 0; j < height; ++j) {
            const uint8* row = src + j * stride;
            int idx = 0;

            for (; idx + 8 <= width; idx+=8) {
                float vals[8];
                for (int k = 0; k < 8; ++k) {
                    vals[k] = row[(idx + k) * target_channels + i] * norm_factor;
                }
                const reg v = avx2::load(vals);
                store(&output.at(0, i, j, idx), v);
            }
            if (idx < 8) {
                const int remain = width - idx;
                float vals[8] = {};
                for (int k = 0; k < remain; ++k) {
                    vals[k] = row[(idx + k) * target_channels + i] * norm_factor;
                }
                const reg v = avx2::load(vals);
                store_partial(&output.at(0, i, j, idx), v, remain);
            }
        }
    }
    if (mean || std) {
        apply(output, mean.value_or(std::array<float,3>{0,0,0}), std.value_or(std::array<float,3>{1,1,1}));
    }

    stbi_image_free(data);
    return output;
}

tensor ImageNetLoader::imagenet(const std::string &path) {
    return load(path);
}

void ImageNetLoader::apply(tensor &x, const std::array<float, 3> &mean, const std::array<float, 3> &std) {
    const reg v_mean[3] = {broadcast(mean[0]), broadcast(mean[1]), broadcast(mean[2])};
    const reg v_std[3]  = {broadcast(std[0]),  broadcast(std[1]),  broadcast(std[2])};

    const int C = x.channel();
    const int HW = x.height() * x.width();
    const int block_per_channel = HW / 8;

    for (int c = 0; c < C; ++c) {
        const size_t block_offset = static_cast<size_t>(c) * block_per_channel;
        for (int i = 0; i < block_per_channel; ++i) {
            const size_t idx = block_offset + i;
            reg v = avx2::load(&x.dataIdx(idx)[0]);
            v = sub(v, v_mean[c]);
            v = div(v, v_std[c]);
            store(&x.dataIdx(idx)[0], v);
        }
    }
}
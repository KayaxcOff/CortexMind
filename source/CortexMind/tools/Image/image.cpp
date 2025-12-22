//
// Created by muham on 13.12.2025.
//

#include "CortexMind/tools/Image/image.hpp"
#include <CortexMind/framework/Tools/Debug/catch.hpp>
#include <CortexMind/framework/Tools/DataTypes/type.hpp>
#include <CortexMind/framework/Core/AVX/funcs.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <STB/stb_image.h>

#include <iostream>

using namespace cortex::tools;
using namespace cortex::_fw;
using namespace cortex::_fw::avx2;
using namespace cortex;

tensor ImageNetLoader::imagenet(const std::string &path) {
    return load(path);
}

tensor ImageNetLoader::load(const std::string &path, const bool normalize, const std::optional<std::array<float, 3> > &mean, const std::optional<std::array<float, 3> > &std, const int target_channels) {
    tensor output;

    int width, height, orig_channels;
    uint8* data = stbi_load(path.c_str(), &width, &height, &orig_channels, target_channels);

    if (!data) {
        CXM_ASSERT(true, "Failed to load image: " + path);
    }

    output.allocate(1, target_channels, height, width);

    const float norm_factor = normalize ? (1.0f / 255.0f) : 1.0f;
    const int C = target_channels;
    //const int HW = width * height;

    const reg v_mean[3] = {
        broadcast(mean.value_or(std::array<float, 3>{0, 0, 0})[0]),
        broadcast(mean.value_or(std::array<float, 3>{0, 0, 0})[1]),
        broadcast(mean.value_or(std::array<float, 3>{0, 0, 0})[2])
    };

    const reg v_std[3] = {
        broadcast(std.value_or(std::array<float, 3>{1, 1, 1})[0]),
        broadcast(std.value_or(std::array<float, 3>{1, 1, 1})[1]),
        broadcast(std.value_or(std::array<float, 3>{1, 1, 1})[2])
    };

    //float* tensor_ptr = output.raw_ptr(0);
    //const size_t stride = C;
    //const size_t total_size = static_cast<size_t>(C) * height * width;

    for (int c = 0; c < C; ++c) {
        for (int j = 0; j < height; ++j) {
            const uint8* row = data + j * width * C;

            int idx = 0;
            while (idx < width) {
                const int remain = std::min(8, width - idx);
                float vals[8] = {0};

                for (int k = 0; k < remain; ++k)
                    vals[k] = static_cast<float>(row[(idx + k) * C + c]) * norm_factor;

                reg v = avx2::load(vals);
                v = sub(v, v_mean[c]);
                v = div(v, v_std[c]);

                const size_t block_idx = (c * height * width + j * width + idx) / 8;
                const size_t offset_in_block = (c * height * width + j * width + idx) % 8;

                if (remain == 8)
                    store(&output.dataIdx(block_idx)[0] + offset_in_block, v);
                else
                    store_partial(&output.dataIdx(block_idx)[0] + offset_in_block, v, remain);

                idx += remain;
            }
        }
    }

    stbi_image_free(data);
    return output;
}

//
// Created by muham on 2.03.2026.
//

#include "CortexMind/utils/Data/image.hpp"
#include <CortexMind/core/Tools/image.hpp>
#include <CortexMind/core/Tools/err.hpp>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <STB/stb_image_resize2.h>

using namespace cortex::utils;
using namespace cortex::_fw;
using namespace cortex;

VisionModule::VisionModule() = default;

VisionModule::~VisionModule() = default;

tensor VisionModule::load(const string &path, const bool gray_scale) {
    return ImageLoader::load(path, gray_scale);
}

std::vector<tensor> VisionModule::load_folder(const string &path, const bool gray_scale) {
    return ImageLoader::load_folder(path, gray_scale);
}

tensor VisionModule::resize(const tensor &image, const int64 height, const int64 width) {
    CXM_ASSERT(image.ndim() == 3, "VisionModule::resize()", "Image must be {C,H,W}.");

    const int64 C      = image.shape()[0];
    const int64 src_H  = image.shape()[1];
    const int64 src_W  = image.shape()[2];

    std::vector<uint8> src_buf(C * src_H * src_W);
    for (int64 c = 0; c < C; ++c)
        for (int64 h = 0; h < src_H; ++h)
            for (int64 w = 0; w < src_W; ++w)
                src_buf[h * src_W * C + w * C + c] =
                    static_cast<uint8>(image.at(c, h, w) * 255.0f);

    std::vector<uint8> dst_buf(C * height * width);
    stbir_resize_uint8_linear(
        src_buf.data(), static_cast<int>(src_W), static_cast<int>(src_H), 0,
        dst_buf.data(), static_cast<int>(width),  static_cast<int>(height), 0,
        static_cast<stbir_pixel_layout>(C)
    );

    tensor output({C, height, width});
    for (int64 c = 0; c < C; ++c)
        for (int64 h = 0; h < height; ++h)
            for (int64 w = 0; w < width; ++w)
                output.at(c, h, w) = static_cast<float32>(
                    dst_buf[h * width * C + w * C + c]) / 255.0f;

    return output;
}

tensor VisionModule::normalize(const tensor &image, const float32 mean, const float32 std) {
    CXM_ASSERT(image.ndim() == 3, "cortex::utils::VisionModule::normalize()", "Image must be {C,H,W}.");
    CXM_ASSERT(std > 0.0f, "cortex::utils::VisionModule::normalize()", "Std must be positive.");

    return (image - mean) / std;
}

tensor VisionModule::flip_horizontal(const tensor &image) {
    CXM_ASSERT(image.ndim() == 3, "cortex::utils::VisionModule::flip_horizontal()", "Image must be {C,H,W}.");

    const int64 C = image.shape()[0];
    const int64 H = image.shape()[1];
    const int64 W = image.shape()[2];

    tensor output({C, H, W});

    for (int64 c = 0; c < C; ++c)
        for (int64 h = 0; h < H; ++h)
            for (int64 w = 0; w < W; ++w)
                output.at(c, h, W - 1 - w) = image.at(c, h, w);

    return output;
}

tensor VisionModule::to_batch(const std::vector<tensor> &images) {
    CXM_ASSERT(!images.empty(), "cortex::utils::VisionModule::to_batch()", "Empty image list");

    const std::vector<int64> img_shape = images[0].shape();
    CXM_ASSERT(img_shape.size() == 3, "cortex::utils::VisionModule::to_batch()", "Each image must be {C,H,W}.");

    const int64 N = static_cast<int64>(images.size());
    const int64 C = img_shape[0];
    const int64 H = img_shape[1];
    const int64 W = img_shape[2];

    tensor output({N, C, H, W});
    for (int64 i = 0; i < N; ++i) {
        CXM_ASSERT(images[i].shape() == img_shape, "cortex::utils::VisionModule::to_batch()", "All images must have the same shape.");
        for (int64 j = 0; j < C; ++j) {
            for (int64 k = 0; k < H; ++k) {
                for (int64 l = 0; l < W; ++l) {
                    output.at(i, j, k, l) = images[i].at(j, k, l);
                }
            }
        }
    }
    return output;
}

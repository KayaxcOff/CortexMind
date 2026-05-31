//
// Created by muham on 30.05.2026.
//

#include "CortexMind/utility/Image/vm.hpp"
#include <CortexMind/framework/Tools/err.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <algorithm>
#include <vector>

using namespace cortex::_fw::sys;
using namespace cortex::utils;
using namespace cortex;

tensor VisionModule::load(const std::filesystem::path &path, ChannelFormat fmt, const DeviceType device) {
    int w = 0, h = 0, orig_c = 0;
    const int C = static_cast<int32>(fmt);

    stbi_uc* raw = stbi_load(path.string().c_str(), &w, &h, &orig_c, C);
    CXM_ASSERT(raw == nullptr, "Cannot open: " + path.string());

    const int64 iC = C, iH = h, iW = w;
    const auto n = static_cast<size_t>(iC * iH * iW);

    std::vector<float32> chw(n);

    const stbi_uc* src = raw;
    for (int64 y = 0; y < iH; ++y) {
        for (int64 x = 0; x < iW; ++x) {
            for (int64 c = 0; c < iC; ++c) {
                chw[static_cast<size_t>(c * iH * iW + y * iW + x)] = static_cast<float32>(src[static_cast<size_t>((y * iW + x) * iC + c)]);
            }
        }
    }

    stbi_image_free(raw);

    return tensor({iC, iH, iW}, chw.data(), device);
}

void VisionModule::save(const tensor &tensor, const std::filesystem::path &path, const int32 jpeg_quality) {
    CXM_ASSERT(tensor.device() != DeviceType::kHOST, "Tensor must be on HOST.");
    CXM_ASSERT(tensor.ndim() < 2 || tensor.ndim() > 3, "Expected [C,H,W] or [H,W] tensor.");
    CXM_ASSERT(!tensor.is_contiguous(), "Tensor must be contiguous. Call .clone() first.");

    const auto& sh = tensor.shape();
    const int64 C = (tensor.ndim() == 2) ? 1   : sh[0];
    const int64 H = (tensor.ndim() == 2) ? sh[0] : sh[1];
    const int64 W = (tensor.ndim() == 2) ? sh[1] : sh[2];

    const auto n = static_cast<size_t>(H * W * C);
    std::vector<stbi_uc> hwc(n);

    const float32* data = tensor.get();

    for (int64 c = 0; c < C; ++c) {
        for (int64 y = 0; y < H; ++y) {
            for (int64 x = 0; x < W; ++x) {
                const float32 v = data[static_cast<size_t>(c * H * W + y * W + x)];
                hwc[static_cast<size_t>((y * W + x) * C + c)] = static_cast<stbi_uc>(std::clamp(v, 0.0f, 255.0f));
            }
        }
    }

    const std::string ext = path.extension().string();
    int32 ok = 0;

    if (ext == ".png") {
        ok = stbi_write_png(path.string().c_str(), static_cast<int32>(W), static_cast<int32>(H), static_cast<int32>(C), hwc.data(), static_cast<int32>(W * C));
    } else if (ext == ".jpg" || ext == ".jpeg") {
        ok = stbi_write_jpg(path.string().c_str(), static_cast<int32>(W), static_cast<int32>(H), static_cast<int32>(C), hwc.data(), jpeg_quality);
    } else if (ext == ".bmp") {
        ok = stbi_write_bmp(path.string().c_str(), static_cast<int32>(W), static_cast<int32>(H), static_cast<int32>(C), hwc.data());
    } else if (ext == ".tga") {
        ok = stbi_write_tga(path.string().c_str(), static_cast<int32>(W), static_cast<int32>(H), static_cast<int32>(C), hwc.data());
    } else {
        CXM_ASSERT(true, "Unsupported extension: " + ext);
    }

    CXM_ASSERT(ok == 0, "Write failed: " + path.string());
}

tensor VisionModule::normalize(const tensor &tensor, const NormMode mode) {
    switch (mode) {
        case NormMode::kNONE:
            return tensor.clone();

        case NormMode::kUNIT:
            return tensor.div(255.0f);

        case NormMode::kSTANDARDIZE:
            return tensor.div(255.0f).mul(2.0f).sub(1.0f);

        case NormMode::kIMAGENET: {
            CXM_ASSERT(tensor.shape()[0] != 3, "kIMAGENET requires an RGB (C=3) tensor.");
            static const std::vector mean = {0.485f, 0.456f, 0.406f};
            static const std::vector std  = {0.229f, 0.224f, 0.225f};
            return normalize(tensor, mean, std);
        }
    }
    return tensor.clone();
}

tensor VisionModule::normalize(const tensor &t, const std::vector<float32> &mean, const std::vector<float32> &std) {
    CXM_ASSERT(t.ndim() != 3, "Expected [C,H,W] tensor.");

    const int64 C = t.shape()[0];
    CXM_ASSERT(static_cast<int64>(mean.size()) != C || static_cast<int64>(std.size())  != C, "mean/std length must equal channel count.");
    CXM_ASSERT(!t.is_contiguous(), "Tensor must be contiguous.");

    const int64 H = t.shape()[1];
    const int64 W = t.shape()[2];
    const auto plane = static_cast<size_t>(H * W);

    tensor output({C, H, W}, t.device());
    const float32* src = t.get();
    float32* dst = output.get();

    for (int64 c = 0; c < C; ++c) {
        const float32  mu  = mean[static_cast<size_t>(c)];
        const float32  sig = std [static_cast<size_t>(c)];
        const float32* s = src + c * static_cast<int64>(plane);
        float32* d = dst + c * static_cast<int64>(plane);

        for (size_t i = 0; i < plane; ++i) {
            d[i] = (s[i] / 255.0f - mu) / sig;
        }
    }
    return output;
}

tensor VisionModule::denormalize(const tensor &t, const NormMode mode) {
    switch (mode) {
        case NormMode::kNONE:
            return t.clone();

        case NormMode::kUNIT:
            // [0, 1] → [0, 255]
            return t.mul(255.0f).clamp(0.0f, 255.0f);

        case NormMode::kSTANDARDIZE:
            // [−1, 1] → [0, 255]
            return t.add(1.0f).div(2.0f).mul(255.0f).clamp(0.0f, 255.0f);

        case NormMode::kIMAGENET:
            CXM_ASSERT(true,
                "[VisionModule::denormalize] kIMAGENET cannot be reversed "
                "without per-channel statistics. "
                "Use the explicit mean/std overload instead.");
    }
    return t.clone();
}
//
// Created by muham on 30.05.2026.
//

#ifndef CORTEXMIND_UTILITY_IMAGE_VM_HPP
#define CORTEXMIND_UTILITY_IMAGE_VM_HPP

#include <CortexMind/framework/Memory/device_type.hpp>
#include <CortexMind/utility/Image/enum.hpp>
#include <filesystem>

namespace cortex::utils {
    /**
     * @brief Stateless image I/O and preprocessing utilities.
     *
     * All heavy arithmetic is delegated to Tensor operations so that
     * SIMD / CUDA paths are used automatically.
     *
     * Tensor layout: [C, H, W], float32.
     *
     * @code
     * auto img  = VisionModule::load("cat.jpg");
     * auto norm = VisionModule::normalize(img, NormMode::kIMAGENET);
     * auto batch = norm.unsqueeze(0);   // [1, 3, H, W]
     * @endcode
     */
    class VisionModule {
    public:
        VisionModule() = delete;

        /**
         * @brief Loads an image from disk and returns a [C, H, W] float32 Tensor.
         *
         * Pixel values are kept in [0, 255].
         * Supported formats: JPEG, PNG, BMP, TGA, PSD, GIF (first frame), HDR, PIC.
         *
         * @param path    Image file path.
         * @param fmt     Channel layout of the returned tensor.
         * @param device  Target device.
         * @return Tensor [C, H, W] in [0, 255].
         */
        [[nodiscard]]
        static tensor load(const std::filesystem::path& path, ChannelFormat fmt = ChannelFormat::kRGB, _fw::sys::DeviceType device = _fw::sys::DeviceType::kHOST);
        /**
         * @brief Saves a [C, H, W] or [H, W] float32 Tensor to disk.
         *
         * The tensor must reside on HOST and values must be in [0, 255].
         * If the tensor was normalized, call @ref denormalize first.
         *
         * @param tensor       Source tensor.
         * @param path         Output path; extension determines format (png/jpg/bmp/tga).
         * @param jpeg_quality JPEG quality [1, 100] (ignored for other formats).
         */
        static void save(const tensor& tensor, const std::filesystem::path& path, int32 jpeg_quality = 90);
        /**
         * @brief Normalizes a [C, H, W] tensor with a preset strategy.
         *
         * Uses Tensor arithmetic so SIMD / CUDA paths are exercised automatically.
         *
         * @param tensor Source tensor in [0, 255].
         * @param mode   Normalization mode.
         * @return Normalized tensor (same shape).
         */
        [[nodiscard]]
        static tensor normalize(const tensor& tensor, NormMode mode = NormMode::kUNIT);
        /**
         * @brief Normalizes with explicit per-channel mean and standard deviation.
         *
         * out[c] = (in[c] / 255 − mean[c]) / std[c]
         *
         * @param t  Source tensor [C, H, W] in [0, 255].
         * @param mean    Per-channel mean (length == C).
         * @param std     Per-channel standard deviation (length == C).
         * @return Normalized tensor.
         */
        [[nodiscard]]
        static tensor normalize(const tensor& t, const std::vector<float32>& mean, const std::vector<float32>& std);

        /**
         * @brief Reverses @ref normalize back to [0, 255].
         *
         * @param t  Previously normalized tensor.
         * @param mode    The mode that was originally applied.
         * @return Tensor in [0, 255].
         */
        [[nodiscard]]
        static tensor denormalize(const tensor& t, NormMode mode);
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_IMAGE_VM_HPP
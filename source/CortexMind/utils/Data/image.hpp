//
// Created by muham on 2.03.2026.
//

#ifndef CORTEXMIND_UTILS_DATA_IMAGE_HPP
#define CORTEXMIND_UTILS_DATA_IMAGE_HPP

#include <CortexMind/tools/params.hpp>
#include <vector>

namespace cortex::utils {
    class VisionModule {
    public:
        VisionModule();
        ~VisionModule();

        [[nodiscard]]
        static tensor load(const string& path, bool gray_scale = false);
        [[nodiscard]]
        static std::vector<tensor> load_folder(const string& path, bool gray_scale = false);
        [[nodiscard]]
        static tensor resize(const tensor& image, int64 height, int64 width);
        [[nodiscard]]
        static tensor normalize(const tensor& image, float32 mean = 0.5f, float32 std  = 0.5f);
        [[nodiscard]]
        static tensor flip_horizontal(const tensor& image);
        [[nodiscard]]
        static tensor to_batch(const std::vector<tensor>& images);
    };
} // namespace cortex::utils

#endif //CORTEXMIND_UTILS_DATA_IMAGE_HPP
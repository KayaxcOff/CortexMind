//
// Created by muham on 13.12.2025.
//

#ifndef CORTEXMIND_IMAGE_HPP
#define CORTEXMIND_IMAGE_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <optional>
#include <array>
#include <string>

namespace cortex::tools {
    class ImageNetLoader {
    public:
        ImageNetLoader() = default;
        ~ImageNetLoader() = default;

       static tensor imagenet(const std::string& path);
    private:
        static tensor load(const std::string& path, bool normalize = true, const std::optional<std::array<float, 3>>& mean = std::nullopt, const std::optional<std::array<float, 3>>& std = std::nullopt, int target_channels = 3);

        static void apply(tensor& x, const std::array<float, 3>& mean, const std::array<float, 3>& std);
    };
}

#endif //CORTEXMIND_IMAGE_HPP
//
// Created by muham on 13.12.2025.
//

#ifndef CORTEXMIND_TYPE_HPP
#define CORTEXMIND_TYPE_HPP

#include <string>
#include <array>
#include <optional>

namespace cortex::_fw {
    struct ImageInfo {
        int width{0};
        int height{0};
        int org_channels{0};
        int load_channels{0};
        std::string path;
        bool is_hdr{false};
    };

    struct TensorSpec {
        int batch{1};
        int channels{3};
        int height{0};
        int width{0};

        bool normalize{true};
        std::optional<std::array<float, 3>> mean;
        std::optional<std::array<float, 3>> std;

        [[nodiscard]] bool valid() const noexcept {
            return batch > 0 && channels > 0 && height > 0 && width > 0;
        }

        [[nodiscard]] size_t total_elements() const noexcept {
            return static_cast<size_t>(batch) * channels * height * width;
        }

        static TensorSpec from_image_info(const ImageInfo& info, const int target_channels = 3) {
            TensorSpec spec;
            spec.width = info.width;
            spec.height = info.height;
            spec.channels = target_channels;
            return spec;
        }
    };
}

#endif //CORTEXMIND_TYPE_HPP
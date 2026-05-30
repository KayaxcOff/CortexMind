//
// Created by muham on 30.05.2026.
//

#ifndef CORTEXMIND_UTILITY_IMAGE_ENUM_HPP
#define CORTEXMIND_UTILITY_IMAGE_ENUM_HPP

#include <CortexMind/tools/types.hpp>

namespace cortex::utils {
    /**
     * @brief Desired channel layout when loading an image.
     */
    enum class ChannelFormat : int32 {
        kGRAY = 1,
        kRGB  = 3,
        kRGBA = 4,
    };

    /**
     * @brief Pixel normalization strategy.
     *
     * | Mode         | Formula                        | Output range |
     * |--------------|--------------------------------|--------------|
     * | kNONE        | x                              | [0, 255]     |
     * | kUNIT        | x / 255                        | [0, 1]       |
     * | kSTANDARDIZE | x / 255 * 2 − 1               | [−1, 1]      |
     * | kIMAGENET    | (x/255 − mean) / std  (RGB)    | per-channel  |
     */
    enum class NormMode {
        kNONE,
        kUNIT,
        kSTANDARDIZE,
        kIMAGENET,
    };
} //namespace cortex::utils

#endif //CORTEXMIND_UTILITY_IMAGE_ENUM_HPP
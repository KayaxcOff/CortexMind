//
// Created by muham on 12.12.2025.
//

#ifndef CORTEXMIND_IMG_HPP
#define CORTEXMIND_IMG_HPP

#include <cstdint>

namespace cortex::_fw {
    struct FileHeader {
        uint16_t type;
        uint32_t size;
        uint16_t reserved1;
        uint16_t reserved2;
        uint32_t offset;
    };

    struct ImageHeader {
        uint32_t size;
        int32_t  width;
        int32_t  height;
        uint16_t planes;
        uint16_t bitCount;
        uint32_t compression;
        uint32_t sizeImage;
        int32_t  xPerMeter;
        int32_t  yPerMeter;
        uint32_t clrUsed;
        uint32_t clrImportant;
    };
}

#endif //CORTEXMIND_IMG_HPP
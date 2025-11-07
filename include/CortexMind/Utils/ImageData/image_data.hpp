//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_IMAGE_DATA_HPP
#define CORTEXMIND_IMAGE_DATA_HPP

#include <string>
#include <vector>
#include <CortexMind/Utils/MathTools/vector/vector.hpp>

namespace cortex::tools {
    class MindImage {
    public:
        explicit MindImage();

        std::vector<math::MindVector> transformPixels(const std::string &_path);
    private:
        std::string path;
    };
}

#endif //CORTEXMIND_IMAGE_DATA_HPP
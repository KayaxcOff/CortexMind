//
// Created by muham on 3.12.2025.
//

#ifndef CORTEXMIND_IMAGE_HPP
#define CORTEXMIND_IMAGE_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <string>

namespace cortex::tools {
    class MindImage {
    public:
        MindImage();
        ~MindImage();

        tensor InitImage(std::string _path);
    };
}

#endif //CORTEXMIND_IMAGE_HPP
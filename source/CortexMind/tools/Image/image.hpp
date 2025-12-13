//
// Created by muham on 13.12.2025.
//

#ifndef CORTEXMIND_IMAGE_HPP
#define CORTEXMIND_IMAGE_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::tools {
    class TensorImg {
    public:
        TensorImg();
        ~TensorImg();

        tensor load();
    };
}

#endif //CORTEXMIND_IMAGE_HPP
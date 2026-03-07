//
// Created by muham on 2.03.2026.
//

#ifndef CORTEXMIND_CORE_TOOLS_IMAGE_HPP
#define CORTEXMIND_CORE_TOOLS_IMAGE_HPP

#include <CortexMind/tools/params.hpp>
#include <vector>

namespace cortex::_fw {
    struct ImageLoader {
        static tensor load(const string& path, bool gray_scale = false);
        static std::vector<tensor> load_folder(const string& path, bool gray_scale = false);
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_TOOLS_IMAGE_HPP
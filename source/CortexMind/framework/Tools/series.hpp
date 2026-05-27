//
// Created by muham on 27.05.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP
#define CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP

#include <CortexMind/framework/Tools/types.hpp>
#include <vector>

namespace cortex::_fw {
    class Series {
    public:
        explicit Series(std::vector<f32>& data);
        ~Series();

        void normalize(f32 value);
        void scale();
        [[nodiscard]]
        std::vector<f32>& data();
    private:
        std::vector<f32>& m_data;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_TOOLS_SERIES_HPP
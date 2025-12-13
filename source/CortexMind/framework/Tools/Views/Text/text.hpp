//
// Created by muham on 12.12.2025.
//

#ifndef CORTEXMIND_TEXT_HPP
#define CORTEXMIND_TEXT_HPP

#include <string>
#include <vector>

namespace cortex::_fw {
    class TensorText {
    public:
        TensorText();
        ~TensorText() = default;

        void initialize(const std::string& filename);

    private:
        std::vector<std::vector<std::string>> texts;
        int rowIdx, colIdx;
    };
}

#endif //CORTEXMIND_TEXT_HPP
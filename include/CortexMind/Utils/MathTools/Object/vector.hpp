//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_VECTOR_HPP
#define CORTEXMIND_VECTOR_HPP

#include <vector>

namespace cortex::vecs {
    class MindVector {
    public:
        MindVector();
        ~MindVector();
    private:
        std::vector<double> matrix;
    };
}

#endif //CORTEXMIND_VECTOR_HPP
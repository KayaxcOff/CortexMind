//
// Created by muham on 6.12.2025.
//

#ifndef CORTEXMIND_OPS_HPP
#define CORTEXMIND_OPS_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex {
    typedef struct MindMath {
        static float mean(const tensor &in) {
            float sum = 0;
            for (auto& item : in.data()) {
                for (const auto& it : item) {
                    sum += it;
                }
            }
            return sum / in.size();
        }

        static float variance(const tensor &in) {
            float sum = 0;
            for (auto& item : in.data()) {
                for (const auto& it : item) {
                    sum += pow(it - sum, 2);
                }
            }
            return sum / in.size();
        }
    } ops;
}

#endif //CORTEXMIND_OPS_HPP
//
// Created by muham on 11.12.2025.
//

#ifndef CORTEXMIND_CAST_HPP
#define CORTEXMIND_CAST_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::tools {
    class TensorFactory {
    public:
        TensorFactory() = default;
        ~TensorFactory() = default;

        template<typename  T>
        static tensor cast(T value) {
            const auto val = static_cast<float>(value);
            tensor output;

            output.allocate(1, 1, 1, 1);

            for (auto& item : output.data()) {
                item.fill(val);
            }
            return output;
        }
    };
}

#endif //CORTEXMIND_CAST_HPP
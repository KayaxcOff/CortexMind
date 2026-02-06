//
// Created by muham on 6.02.2026.
//

#ifndef CORTEXMIND_CORE_NET_LAYER_HPP
#define CORTEXMIND_CORE_NET_LAYER_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/core/Tools/ref.hpp>
#include <vector>

namespace cortex::_fw {
    class Layer {
    public:
        Layer() = default;

        virtual
        ~Layer() = default;

        [[nodiscard]] virtual
        tensor forward(const tensor& input) = 0;

        [[nodiscard]] virtual
        string config() = 0;

        [[nodiscard]] virtual
        std::vector<ref<tensor>> parameters() = 0;

        [[nodiscard]] virtual
        std::vector<ref<tensor>> gradients() = 0;

        [[nodiscard]] virtual
        boolean is_train() = 0;

        virtual
        void set_train(boolean _is_train) = 0;
    protected:
        tensor weights, bias;

        boolean flag;

        string name;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_LAYER_HPP
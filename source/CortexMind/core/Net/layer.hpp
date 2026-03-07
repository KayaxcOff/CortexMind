//
// Created by muham on 25.02.2026.
//

#ifndef CORTEXMIND_CORE_NET_LAYER_HPP
#define CORTEXMIND_CORE_NET_LAYER_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/core/Tools/ref.hpp>
#include <vector>

namespace cortex::_fw {
    class Layer {
    public:
        explicit
        Layer(bool _train_flag, string info);
        Layer(const Layer&) = delete;
        Layer(Layer&&) = default;
        virtual ~Layer() = default;

        [[nodiscard]]
        virtual tensor forward(tensor& input) = 0;
        [[nodiscard]]
        virtual std::vector<ref<tensor>> parameters() = 0;
        [[nodiscard]]
        virtual std::vector<ref<tensor>> gradients() = 0;

        [[nodiscard]]
        bool is_train() const;
        void set_train(bool _train);
        [[nodiscard]]
        string config();

        Layer& operator=(const Layer&) = delete;
        Layer& operator=(Layer&&) = default;
    protected:
        bool flag;
        tensor last_input;
    private:
        string info;
    };
} // namespace cortex::_fw

#endif //CORTEXMIND_CORE_NET_LAYER_HPP
//
// Created by muham on 29.12.2025.
//

#ifndef CORTEXMIND_CORE_NET_LAYER_HPP
#define CORTEXMIND_CORE_NET_LAYER_HPP

#include <core/Params/params.hpp>

#include <vector>
#include <string>

namespace cortex::_fw {
    class Layer {
    public:
        Layer() = default;
        virtual ~Layer() = default;

        virtual tensor forward(const tensor& input) = 0;
        [[nodiscard]] virtual std::string config() const = 0;
        [[nodiscard]] virtual std::vector<tensor> parameters() const = 0;
    protected:
        tensor weights;
        tensor bias;
    };
} // namespace cortex::_fw

#endif // CORTEXMIND_CORE_NET_LAYER_HPP
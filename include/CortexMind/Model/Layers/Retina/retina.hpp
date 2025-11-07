//
// Created by muham on 3.11.2025.
//

#ifndef CORTEXMIND_RETINA_HPP
#define CORTEXMIND_RETINA_HPP

#include <CortexMind/Model/Layers/layer.hpp>
#include <string>
#include <STB/stb_image.h>

namespace cortex::layer {
    class Retina final : public Layer {
    public:
        explicit Retina(std::string _path);
        ~Retina() override;

        math::MindVector forward(const math::MindVector &input) override;
        math::MindVector backward(const math::MindVector &grad_output) override;
        void update(double lr) override;
    private:
        stbi_uc *data{};
        std::string path;
    };
}

#endif //CORTEXMIND_RETINA_HPP
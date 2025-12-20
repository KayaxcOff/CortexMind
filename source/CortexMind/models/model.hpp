//
// Created by muham on 20.12.2025.
//

#ifndef CORTEXMIND_BASEMODEL_HPP
#define CORTEXMIND_BASEMODEL_HPP

#include <CortexMind/framework/Params/params.hpp>
#include <vector>

namespace cortex::tin {
    class Model {
    public:
        Model() = default;
        virtual ~Model() = default;

        virtual void fit(std::vector<tensor>& X, std::vector<tensor>& Y) = 0;
        virtual tensor predict(const tensor& pred) = 0;
    };
}

#endif //CORTEXMIND_BASEMODEL_HPP
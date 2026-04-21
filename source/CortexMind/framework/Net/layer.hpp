//
// Created by muham on 21.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_LAYER_HPP
#define CORTEXMIND_FRAMEWORK_NET_LAYER_HPP

#include <CortexMind/tools/params.hpp>
#include <string>
#include <vector>

namespace cortex::_fw {
    class LayerBase {
    public:
        explicit LayerBase(std::string  name, boolean _train_flag = true);
        virtual ~LayerBase();

        [[nodiscard]]
        virtual tensor forward(tensor& input) = 0;
        [[nodiscard]]
        virtual std::vector<tensor> getWeights() = 0;
        [[nodiscard]]
        virtual std::vector<tensor> getGradients() = 0;

        void TrainMode();
        void EvalMode();
        [[nodiscard]]
        const std::string& getName() const;
    private:
        std::string kName;
        boolean kTrainFlag;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_LAYER_HPP
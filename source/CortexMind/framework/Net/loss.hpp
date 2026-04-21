//
// Created by muham on 21.04.2026.
//

#ifndef CORTEXMIND_FRAMEWORK_NET_LOSS_HPP
#define CORTEXMIND_FRAMEWORK_NET_LOSS_HPP

#include <CortexMind/tools/params.hpp>
#include <string>

namespace cortex::_fw {
    class LossBase {
    public:
        explicit LossBase(std::string name);
        virtual ~LossBase();

        [[nodiscard]]
        virtual tensor forward(tensor& prediction, tensor& target) = 0;

        [[nodiscard]]
        const std::string& getName() const;
    private:
        std::string kName;
    };
} //namespace cortex::_fw

#endif //CORTEXMIND_FRAMEWORK_NET_LOSS_HPP
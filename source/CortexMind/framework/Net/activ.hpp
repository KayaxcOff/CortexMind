//
// Created by muham on 5.12.2025.
//

#ifndef CORTEXMIND_ACTIV_HPP
#define CORTEXMIND_ACTIV_HPP

namespace cortex::_fw {
    class Activation {
    public:
        Activation() = default;
        virtual ~Activation() = default;

        virtual tensor forward(const tensor& input) = 0;
        virtual tensor backward(const tensor& input) = 0;
    };
}

#endif //CORTEXMIND_ACTIV_HPP
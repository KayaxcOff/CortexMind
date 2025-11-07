//
// Created by muham on 4.11.2025.
//

#ifndef CORTEXMIND_OPTIMIZER_HPP
#define CORTEXMIND_OPTIMIZER_HPP

namespace cortex::optim {
    class Optimizer {
    public:
        virtual ~Optimizer();

        virtual void step();
    };
}

#endif //CORTEXMIND_OPTIMIZER_HPP
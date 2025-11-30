//
// Created by muham on 30.11.2025.
//

#ifndef CORTEXMIND_KERNEL_HPP
#define CORTEXMIND_KERNEL_HPP

#include <CortexMind/framework/Params/params.hpp>

namespace cortex::tools {
    class MindKernel {
    public:
        MindKernel(size_t _size, size_t _in, size_t _out);
        ~MindKernel();

        tensor& getWeights() { return this->weights; }
        tensor& getBias() { return this->biases; }
        tensor& getGradWeights() { return this->gradWeights; }
        tensor& getGradBias() { return this->gradBias; }

        size_t getSize() const { return this->size; }
        size_t getInFeat() const { return this->in; }
        size_t getOutFeat() const { return this->out; }

        void zero_grad();
    private:
        tensor weights;
        tensor biases;

        tensor gradWeights;
        tensor gradBias;

        size_t size;
        size_t in, out;

        void Init();
    };
}

#endif //CORTEXMIND_KERNEL_HPP
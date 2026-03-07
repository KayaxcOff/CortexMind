//
// Created by muham on 4.03.2026.
//

#ifndef CORTEXMIND_DATASET_MANAGER_DATA_HPP
#define CORTEXMIND_DATASET_MANAGER_DATA_HPP

#include <CortexMind/tools/params.hpp>

namespace cortex::ds {
    class TensorSet {
    public:
        TensorSet(int32 d_type, bool requires_grad);
        ~TensorSet();

        tensor load();
    private:
        tensor data;
        int32 d_type;
        bool requires_grad;
    };
} // namespace cortex::ds

#endif //CORTEXMIND_DATASET_MANAGER_DATA_HPP
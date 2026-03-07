//
// Created by muham on 1.03.2026.
//

#ifndef CORTEXMIND_TOOLS_DEFAULTS_HPP
#define CORTEXMIND_TOOLS_DEFAULTS_HPP

#include <CortexMind/tools/params.hpp>
#include <CortexMind/core/Tools/defaults.hpp>

namespace cortex {
    inline
    int64 epochs     = CXM_EPOCH;
    inline
    int64 max_epochs = CXM_MAX_EPOCH;
    inline
    int64 batch      = CXM_BATCH;
    inline
    int32 exit       = CXM_EXIT;
    inline
    int32 err_exit   = CXM_ERR_EXIT;
    inline
    int32 tensors    = CXM_TENSOR;
    inline
    int32 pixels     = CXM_PIXEL;
    inline
    int32 csv        = CXM_CSV;
} // namespace cortex

#endif //CORTEXMIND_TOOLS_DEFAULTS_HPP
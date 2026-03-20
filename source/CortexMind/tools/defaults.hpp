//
// Created by muham on 16.03.2026.
//

#ifndef CORTEXMIND_CORE_DEFAULTS_HPP
#define CORTEXMIND_CORE_DEFAULTS_HPP

#include <CortexMind/core/Tools/defaults.hpp>
#include <CortexMind/tools/params.hpp>

namespace cortex {
    inline int32 safety_exit    = CXM_EXIT;
    inline int32 err_exit       = CXM_ERR_EXIT;
    inline int32 epochs         = CXM_DEFAULT_EPOCH;
    inline int32 max_epochs     = CXM_DEFAULT_MAX_EPOCH;
} // namespace cortex

#endif //CORTEXMIND_CORE_DEFAULTS_HPP
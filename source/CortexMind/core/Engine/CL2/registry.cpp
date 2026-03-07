//
// Created by muham on 22.02.2026.
//

#include "CortexMind/core/Engine/CL2/funcs.hpp"
#include <CortexMind/core/Tools/file_system.hpp>

using namespace cortex::_fw::cl2;
using namespace cortex::_fw;

registry& registry::get() {
    static registry instance(resolve_kernel_dir());
    return instance;
}

fs::path registry::resolve_kernel_dir() {
#ifdef CORTEX_KERNEL_DIR
    return fs::path(CORTEX_KERNEL_DIR);
#else
    return fs::current_path() / "kernels";
#endif
}

registry::registry(const fs::path& kernel_dir) {
    CXM_ASSERT(fs::exists(kernel_dir), "cortex::_fw::cl2::registry::registry()", "Can't found kernel: " + kernel_dir.string());

    this->elem_   = std::make_unique<program>(kernel_dir / "elementwise.cl");
    this->matmul_ = std::make_unique<program>(kernel_dir / "matmul.cl");
    this->reduce_ = std::make_unique<program>(kernel_dir / "reduce.cl");
}
//
// Created by muham on 18.04.2026.
//

#include "CortexMind/framework/Tensor/tensor.hpp"
#if CXM_IS_CUDA_AVAILABLE
    #include <CortexMind/framework/Memory/transform.hpp>
#else //#if CXM_IS_CUDA_AVAILABLE
    #include <cstring>
#endif //#if CXM_IS_CUDA_AVAILABLE #else
#include <iostream>
#include <functional>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

namespace cortex::_fw {
    std::ostream& operator<<(std::ostream& os, const MindTensor& tensor) {
        const auto& shape = tensor.storage_->shape;
        const size_t numel = tensor.numel();

        std::vector<f32> host_data(numel);

        if (tensor.device() == sys::deviceType::host) {
            std::memcpy(host_data.data(), tensor.get(), numel * sizeof(f32));
        }
        #if CXM_IS_CUDA_AVAILABLE
        else {
            transform<f32>::download(host_data.data(), tensor.get(), numel);
        }
        #endif //#if CXM_IS_CUDA_AVAILABLE

        std::function<void(size_t, size_t, size_t)> print_dim =
            [&](const size_t dim, const size_t offset, const size_t indent) {
                if (dim == shape.size() - 1) {
                    os << "[";
                    for (i64 i = 0; i < shape[dim]; ++i) {
                        os << host_data[offset + i];
                        if (i < shape[dim] - 1) os << ", ";
                    }
                    os << "]";
                } else {
                    os << "[";
                    const size_t stride = tensor.storage_->stride[dim];
                    for (i64 i = 0; i < shape[dim]; ++i) {
                        if (i > 0) {
                            os << ",\n";
                            for (size_t s = 0; s <= indent; ++s) os << " ";
                        }
                        print_dim(dim + 1, offset + i * stride, indent + 1);
                    }
                    os << "]";
                }
        };

        print_dim(0, tensor.storage_->offset, 0);
        os << "\n";

        return os;
    }
} //namespace cortex::_fw
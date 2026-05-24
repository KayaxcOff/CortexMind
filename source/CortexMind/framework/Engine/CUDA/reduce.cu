//
// Created by muham on 9.05.2026.
//

#include "CortexMind/framework/Engine/CUDA/reduce.cuh"
#include <CortexMind/framework/Engine/CUDA/Kernels/reduce.cuh>
#include <CortexMind/framework/Tools/cuda.cuh>
#include <CortexMind/framework/Tools/err.hpp>
#include <CortexMind/runtime/provider.cuh>
#include <cmath>

using namespace cortex::_fw::cuda;
using namespace cortex::_fw;

ReduceOp::ReduceOp() {
    host::allocate(reinterpret_cast<void **>(&this->host_output), sizeof(f32), CXM_HOST_ALLOC_MAPPED);
    map(reinterpret_cast<void **>(&this->cuda_output), this->host_output);

    host::allocate(reinterpret_cast<void**>(&this->host_index), sizeof(i32), CXM_HOST_ALLOC_MAPPED);
    map(reinterpret_cast<void**>(&this->cuda_index), this->host_index);
}

ReduceOp::~ReduceOp() {
    host::free(this->host_output);
    host::free(this->host_index);
}

f32 ReduceOp::sum(const f32 *x, size_t N) {
    CXM_CUDA_ASSERT(memset<f32>(this->cuda_output, 0, 1));

    constexpr size_t shared_mem = (BLOCK_SIZE_1D / WARP_SIZE) * sizeof(f32);
    kernels::ReduceSum<<<grid1d(N), BLOCK_SIZE_1D, shared_mem>>>(x, this->cuda_output, N);

    DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::mean(const f32* x, const size_t N) {
    return this->sum(x, N) / static_cast<f32>(N);
}

f32 ReduceOp::var(const f32 *x, size_t N) {
    const f32 mu = this->mean(x, N);
    CXM_CUDA_ASSERT(memset<f32>(this->cuda_output, 0, 1));

    constexpr size_t shared_mem = (BLOCK_SIZE_1D / WARP_SIZE) * sizeof(f32);
    kernels::ReduceVar<<<grid1d(N), BLOCK_SIZE_1D, shared_mem>>>(x, mu, this->cuda_output, N);

    DeviceSynchronize();
    return *this->host_output / static_cast<f32>(N);
}

f32 ReduceOp::stdv(const f32* x, const size_t N) {
    return std::sqrt(this->var(x, N));
}

f32 ReduceOp::min(const f32* x, const size_t N) const {
    const cublasStatus_t stat = cublasIsamin(runtime::Provider::instance().handle, static_cast<i32>(N), x, 1, this->cuda_index);
    CXM_ASSERT(stat != CUBLAS_STATUS_SUCCESS, "cublasIsamin failed");
    DeviceSynchronize();
    const i32 idx = *this->host_index - 1;
    f32 result;
    memcpy(&result, x + idx, sizeof(f32), CXM_DEVICE_TO_HOST);
    return result;
}

f32 ReduceOp::max(const f32* x, const size_t N) const {
    const cublasStatus_t stat = cublasIsamax(runtime::Provider::instance().handle, static_cast<i32>(N), x, 1, this->cuda_index);
    CXM_ASSERT(stat != CUBLAS_STATUS_SUCCESS, "cublasIsamax failed");
    DeviceSynchronize();
    const i32 idx = *this->host_index - 1;
    f32 result;
    memcpy(&result, x + idx, sizeof(f32), CXM_DEVICE_TO_HOST);
    return result;
}

f32 ReduceOp::norm1(const f32* x, const size_t N) const {
    const cublasStatus_t stat = cublasSasum(runtime::Provider::instance().handle, static_cast<i32>(N), x, 1, this->cuda_output);
    CXM_ASSERT(stat != CUBLAS_STATUS_SUCCESS, "cublasIsasum failed");
    DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::norm2(const f32* x, const size_t N) const {
    const cublasStatus_t stat = cublasSnrm2(runtime::Provider::instance().handle, static_cast<i32>(N), x, 1, this->cuda_output);
    CXM_ASSERT(stat != CUBLAS_STATUS_SUCCESS, "cublasSnrm2 failed");
    DeviceSynchronize();
    return *this->host_output;
}

f32 ReduceOp::dot(const f32* Xx, const f32* Xy, const size_t N) const {
    const cublasStatus_t stat = cublasSdot(runtime::Provider::instance().handle,static_cast<i32>(N), Xx, 1, Xy, 1, this->cuda_output);
    CXM_ASSERT(stat != CUBLAS_STATUS_SUCCESS, "cublasSdot failed");
    DeviceSynchronize();
    return *this->host_output;
}

void ReduceOp::mean_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    // CUDA version: For now, copying to CPU, processing, and copying back
    // Optimal implementation would use CUDA kernels
    
    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    std::vector<f32> host_input(total_elements);
    std::vector<f32> host_output(total_elements);
    
    memcpy(host_input.data(), x, total_elements * sizeof(f32), CXM_DEVICE_TO_HOST);
    
    // Use AVX2 version on host
    avx2::reduce::mean_dim(
        host_input.data(),
        host_output.data(),
        shape,
        strides,
        reduce_dims,
        ndim,
        num_reduce_dims
    );

    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }
    
    memcpy(output, host_output.data(), out_size * sizeof(f32), CXM_HOST_TO_DEVICE);
    DeviceSynchronize();
}

void ReduceOp::var_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    std::vector<f32> host_input(total_elements);
    std::vector<f32> host_output(total_elements);
    
    memcpy(host_input.data(), x, total_elements * sizeof(f32), CXM_DEVICE_TO_HOST);
    
    avx2::reduce::var_dim(
        host_input.data(),
        host_output.data(),
        shape,
        strides,
        reduce_dims,
        ndim,
        num_reduce_dims
    );

    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }
    
    memcpy(output, host_output.data(), out_size * sizeof(f32), CXM_HOST_TO_DEVICE);
    DeviceSynchronize();
}

void ReduceOp::stdv_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    std::vector<f32> host_input(total_elements);
    std::vector<f32> host_output(total_elements);
    
    memcpy(host_input.data(), x, total_elements * sizeof(f32), CXM_DEVICE_TO_HOST);
    
    avx2::reduce::std_dim(
        host_input.data(),
        host_output.data(),
        shape,
        strides,
        reduce_dims,
        ndim,
        num_reduce_dims
    );

    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }
    
    memcpy(output, host_output.data(), out_size * sizeof(f32), CXM_HOST_TO_DEVICE);
    DeviceSynchronize();
}

void ReduceOp::min_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    std::vector<f32> host_input(total_elements);
    std::vector<f32> host_output(total_elements);
    
    memcpy(host_input.data(), x, total_elements * sizeof(f32), CXM_DEVICE_TO_HOST);
    
    avx2::reduce::min_dim(
        host_input.data(),
        host_output.data(),
        shape,
        strides,
        reduce_dims,
        ndim,
        num_reduce_dims
    );

    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }
    
    memcpy(output, host_output.data(), out_size * sizeof(f32), CXM_HOST_TO_DEVICE);
    DeviceSynchronize();
}

void ReduceOp::max_dim(
    const f32* __restrict x,
    f32* __restrict output,
    const i64* shape,
    const i64* strides,
    const i64* reduce_dims,
    size_t ndim,
    size_t num_reduce_dims
) {
    size_t total_elements = 1;
    for (size_t d = 0; d < ndim; ++d) {
        total_elements *= static_cast<size_t>(shape[d]);
    }

    std::vector<f32> host_input(total_elements);
    std::vector<f32> host_output(total_elements);
    
    memcpy(host_input.data(), x, total_elements * sizeof(f32), CXM_DEVICE_TO_HOST);
    
    avx2::reduce::max_dim(
        host_input.data(),
        host_output.data(),
        shape,
        strides,
        reduce_dims,
        ndim,
        num_reduce_dims
    );

    size_t out_size = 1;
    for (size_t d = 0; d < ndim; ++d) {
        bool is_reduced = false;
        for (size_t r = 0; r < num_reduce_dims; ++r) {
            if (reduce_dims[r] == static_cast<i64>(d)) {
                is_reduced = true;
                break;
            }
        }
        if (!is_reduced) {
            out_size *= static_cast<size_t>(shape[d]);
        }
    }
    
    memcpy(output, host_output.data(), out_size * sizeof(f32), CXM_HOST_TO_DEVICE);
    DeviceSynchronize();
}
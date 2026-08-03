//
// Created by muham on 3.08.2026.
//

#include "CortexMind/framework/Memory/transform.cuh"
#include <CortexMind/framework/Tools/errors.hpp>
#include <cuda_runtime.h>

using namespace cortex::_fw;

namespace {
    constexpr auto CXM_HOST_TO_DEVICE = cudaMemcpyHostToDevice;
    constexpr auto CXM_DEVICE_TO_HOST = cudaMemcpyDeviceToHost;
    constexpr auto CXM_DEVICE_TO_DEVICE = cudaMemcpyDeviceToDevice;
} //namespace

void transform::upload(void *dst, const void *src, const std::size_t byte) {
    CXM_DEVICE_ASSERT(
        cudaMemcpy(dst, src, byte, CXM_HOST_TO_DEVICE),
        "Data can't upload to device from host"
    );
}

void transform::download(void *dst, const void *src, const std::size_t byte) {
    CXM_DEVICE_ASSERT(
        cudaMemcpy(dst, src, byte, CXM_DEVICE_TO_HOST),
        "Data can't download to host from device"
    );
}

void transform::copy_h2h(void *dst, const void *src, const std::size_t byte) {
    std::memcpy(dst, src, byte);
}

void transform::copy_d2d(void *dst, const void *src, const std::size_t byte) {
    CXM_DEVICE_ASSERT(
        cudaMemcpy(dst, src, byte, CXM_DEVICE_TO_DEVICE),
        "Data can't copy from device to device"
    );
}
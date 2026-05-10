//
// Created by muham on 10.05.2026.
//

#include "CortexMind/framework/Memory/transform.cuh"
#include <CortexMind/framework/Tools/cuda.cuh>
#include <cstring>

using namespace cortex::_fw::sys;
using namespace cortex::_fw;

void transform::upload(f32 *dst, const f32 *src, const size_t count) {
    cuda::memcpy(dst, src, sizeof(f32) * count, cuda::CXM_HOST_TO_DEVICE);
}

void transform::download(f32 *dst, const f32 *src, const size_t count) {
    cuda::memcpy(dst, src, sizeof(f32) * count, cuda::CXM_DEVICE_TO_HOST);
}

void transform::copy_h2h(f32 *dst, const f32 *src, const size_t count) {
    std::memcpy(dst, src, sizeof(f32) * count);
}

void transform::copy_d2d(f32 *dst, const f32 *src, const size_t count) {
    cuda::memcpy(dst, src, sizeof(f32) * count, cuda::CXM_DEVICE_TO_DEVICE);
}
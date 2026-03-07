//
// Created by muham on 21.02.2026.
//

#include "CortexMind/core/Engine/CL2/context.hpp"
#include <CortexMind/core/Tools/err.hpp>
#include <iostream>

using namespace cortex::_fw::cl2;

runtime& runtime::get() {
    static runtime instance;
    return instance;
}

const cl::Context &runtime::context() const {
    return this->m_ctx;
}

const cl::CommandQueue &runtime::queue() const {
    return this->m_queue;
}

const cl::Device &runtime::device() const {
    return this->m_device;
}

runtime::runtime() {
    select_platform(this->m_platform);
    select_device(this->m_platform, this->m_device);

    this->m_ctx = cl::Context(this->m_device);
    this->m_queue = cl::CommandQueue(this->m_ctx, this->m_device, cl::QueueProperties::Profiling);

    std::cout << "[CortexMind CL] Platform : " << this->m_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
    std::cout << "[CortexMind CL] Device   : " << this->m_device.getInfo<CL_DEVICE_NAME>()   << "\n";
}

void runtime::select_platform(::cl::Platform& out) {
    std::vector<::cl::Platform> platforms;
    ::cl::Platform::get(&platforms);

    CXM_ASSERT(!platforms.empty(), "cortex::_fw::cl2::runtime::select_platform()", "Invalid platform.");

    for (auto& item : platforms) {
        const std::string name = item.getInfo<CL_PLATFORM_NAME>();
        if (name.find("Intel") != std::string::npos) { out = item; return; }
    }
    out = platforms[0];
}

void runtime::select_device(const ::cl::Platform& p, ::cl::Device& out) {
    std::vector<::cl::Device> devices;
    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        p.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    }

    CXM_ASSERT(!devices.empty(), "cortex::_fw::cl2::runtime::select_device()", "Invalid device.");
    out = devices[0];
}
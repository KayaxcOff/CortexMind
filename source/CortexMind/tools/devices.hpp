//
// Created by muham on 22.02.2026.
//

#ifndef CORTEXMIND_TOOLS_DEVICES_HPP
#define CORTEXMIND_TOOLS_DEVICES_HPP

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>
#include <iomanip>

namespace cortex {
    inline
    void PrintDevices() {

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        if (platforms.empty()) {
            std::cout << "No OpenCL platforms found.\n";
            return;
        }

        for (size_t i = 0; i < platforms.size(); ++i)
        {
            std::string platformName = platforms[i].getInfo<CL_PLATFORM_NAME>();
            std::cout << "Platform " << i << ": " << platformName << "\n";

            std::vector<cl::Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (size_t j = 0; j < devices.size(); ++j)
            {
                cl::Device& device = devices[j];

                std::string name = device.getInfo<CL_DEVICE_NAME>();
                std::string vendor = device.getInfo<CL_DEVICE_VENDOR>();
                std::string version = device.getInfo<CL_DEVICE_VERSION>();

                const cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();

                std::string typeStr;
                if (type & CL_DEVICE_TYPE_GPU)
                    typeStr = "GPU";
                else if (type & CL_DEVICE_TYPE_CPU)
                    typeStr = "CPU";
                else if (type & CL_DEVICE_TYPE_ACCELERATOR)
                    typeStr = "ACCELERATOR";
                else
                    typeStr = "OTHER";

                const cl_uint computeUnits =
                    device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

                const cl_uint clock =
                    device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();

                const cl_ulong globalMem =
                    device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();

                std::cout << "  Device " << j << " (" << typeStr << ")\n";
                std::cout << "    Name: " << name << "\n";
                std::cout << "    Vendor: " << vendor << "\n";
                std::cout << "    Version: " << version << "\n";
                std::cout << "    Compute Units: " << computeUnits << "\n";
                std::cout << "    Max Clock: " << clock << " MHz\n";
                std::cout << "    Global Memory: "
                          << (globalMem / (1024 * 1024)) << " MB\n";
            }

            std::cout << "-----------------------------------\n";
        }

    }
} // namespace cortex

#endif //CORTEXMIND_TOOLS_DEVICES_HPP
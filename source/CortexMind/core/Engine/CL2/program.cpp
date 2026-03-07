//
// Created by muham on 22.02.2026.
//

#include "CortexMind/core/Engine/CL2/program.hpp"
#include <fstream>
#include <sstream>
#include <utility>

using namespace cortex::_fw::cl2;

program::program(const fs::path &path) : name_(path.filename().string()) {
    CXM_ASSERT(fs::exists(path), "cortex::_fw::cl2::program::program()", "Can't found path: " + path.string());

    std::ifstream file(path);
    CXM_ASSERT(file.is_open(), "cortex::_fw::cl2::program::program()", "Can't open file: " + path.string());

    std::ostringstream ss;
    ss << file.rdbuf();

    this->prog_ = cl::Program(runtime::get().context(), ss.str());
    this->build();
}

program::program(const std::string& source, std::string  name) : name_(std::move(name)) {
    CXM_ASSERT(!source.empty(), "cortex::_fw::cl2::program::program()", "Source is empty.");
    this->prog_ = ::cl::Program(runtime::get().context(), source);
    this->build();
}

void program::build() const {
    const cl_int err = this->prog_.build();

    if (err != CL_SUCCESS) {
        const std::string log = this->prog_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
            runtime::get().device()
        );
        CXM_ASSERT(false, "cortex::_fw::cl2::program::build()", "Compile Error [" + this->name_ + "]:\n" + log);
    }
}

cl::Kernel& program::kernel(const std::string& name) {
    auto it = this->cache_.find(name);
    if (it != this->cache_.end()) return it->second;

    cl_int err;
    cl::Kernel k(this->prog_, name.c_str(), &err);
    CXM_ASSERT(err == CL_SUCCESS, "cortex::_fw::cl2::program::kernel()", "Can't found: " + name);

    this->cache_.emplace(name, std::move(k));
    return this->cache_.at(name);
}

const std::string &program::name() const {
    return this->name_;
}

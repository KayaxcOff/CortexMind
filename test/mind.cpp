//
// Created by muham on 30.11.2025.
//

#include <CortexMind/framework/Params/params.hpp>
#include <CortexMind/framework/Kernel/kernel.hpp>
#include <memory>
#include <iostream>

using namespace cortex;

int main() {
    auto x = tensor(1, 5, 5);

    x.uniform_rand(0.1, 1.0);

    std::cout << std::endl;
    x.print();
    std::cout << std::endl;

    const auto mind_kernel_ = std::make_unique<tools::MindKernel>(1, 2, 3);

    const auto result = mind_kernel_->apply(x);

    std::cout << std::endl;
    result.print();
    std::cout << std::endl;

    return 0;
}
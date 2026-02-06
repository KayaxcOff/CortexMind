//
// Created by muham on 2.02.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    tensor x({3, 3});

    x.uniform_rand(2, 3);

    println("X before flatten");
    x.print();

    println("X after flatten");
    x.flatten().print();

    return 0;
}
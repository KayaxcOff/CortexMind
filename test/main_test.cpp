//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>

using namespace cortex;

int main() {
    const tensor x({3, 3}, cuda);
    x.rand();

    info(x);

    return 0;
}
//
// Created by muham on 7.04.2026.
//

#include <CortexMind/cortexmind.hpp>
#include <iostream>

using namespace cortex;

int main() {

    float data_a[] = {1,2,3,4,5,6};
    float data_b[] = {7,8,9,10,11,12};

    tensor a({2,3}, data_a, device_t::cuda);
    tensor b({3,2}, data_b, device_t::cuda);

    std::cout << a.dot(b) << std::endl;

    return 0;
}
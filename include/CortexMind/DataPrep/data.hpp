//
// Created by muham on 8.11.2025.
//

#ifndef CORTEXMIND_DATA_HPP
#define CORTEXMIND_DATA_HPP

#include <CortexMind/Utils/params.hpp>
#include <vector>

namespace cortex::prep {
    class DataGen {
    public:
        DataGen();
        ~DataGen();

        std::vector<tensor> float_to_tensor(const std::vector<float32>& data);
        std::vector<tensor> int_to_tensor(const std::vector<int32>& data);
    private:
        std::vector<int32> int_data_;
        std::vector<float32> float_data_;
    };
}

#endif //CORTEXMIND_DATA_HPP
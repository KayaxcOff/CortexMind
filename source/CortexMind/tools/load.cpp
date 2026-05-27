//
// Created by muham on 27.05.2026.
//

#include "CortexMind/tools/load.hpp"

using namespace cortex::utils;

DataFrame cortex::load(const std::string& file_name) {
    DataFrame df(file_name);
    return df;
}
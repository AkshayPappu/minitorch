#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>

namespace minitorch {

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::Float32: return 4;
        case DType::Float64: return 8;
        case DType::Int32:   return 4;
        case DType::Int64:   return 8;
    }
    throw std::runtime_error("Unknown dtype");
}

inline std::string dtype_name(DType dt) {
    switch (dt) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32:   return "int32";
        case DType::Int64:   return "int64";
    }
    throw std::runtime_error("Unknown dtype");
}

enum class Device {
    CPU,
    CUDA,
};

inline std::string device_name(Device d) {
    switch (d) {
        case Device::CPU:  return "cpu";
        case Device::CUDA: return "cuda";
    }
    throw std::runtime_error("Unknown device");
}

} // namespace minitorch

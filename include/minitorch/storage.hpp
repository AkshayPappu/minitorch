#pragma once

#include <memory>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace minitorch {

class Storage {
public:
    Storage() : data_(nullptr), size_(0) {}

    explicit Storage(size_t size)
        : data_(new float[size], std::default_delete<float[]>()), size_(size) {}

    Storage(size_t size, float value)
        : data_(new float[size], std::default_delete<float[]>()), size_(size) {
        for (size_t i = 0; i < size; ++i)
            data_.get()[i] = value;
    }

    static Storage from_blob(float* ptr, size_t size) {
        Storage s;
        s.data_ = std::shared_ptr<float[]>(ptr, [](float*) {});
        s.size_ = size;
        return s;
    }

    float* data_ptr() const { return data_.get(); }
    size_t size() const { return size_; }
    bool is_valid() const { return data_ != nullptr; }
    long use_count() const { return data_.use_count(); }

    Storage clone() const {
        Storage s(size_);
        std::memcpy(s.data_.get(), data_.get(), size_ * sizeof(float));
        return s;
    }

private:
    std::shared_ptr<float[]> data_;
    size_t size_;
};

} // namespace minitorch

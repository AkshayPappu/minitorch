#include <gtest/gtest.h>
#include <minitorch/tensor.hpp>

using namespace minitorch;

TEST(TensorTest, DefaultConstruction) {
    Tensor t;
    SUCCEED();
}

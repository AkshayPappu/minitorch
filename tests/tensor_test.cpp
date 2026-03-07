#include <gtest/gtest.h>
#include <minitorch/tensor.hpp>
#include <cmath>
#include <limits>

using namespace minitorch;

static constexpr float EPS = 1e-5f;

// ════════════════════════════════════════════════════════════════
// Phase 1: Core Metadata & Storage Access
// ════════════════════════════════════════════════════════════════

TEST(Phase1, DefaultConstruction) {
    Tensor t;
    EXPECT_EQ(t.dim(), 0);
    EXPECT_EQ(t.numel(), 1);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(t.dtype(), DType::Float32);
    EXPECT_EQ(t.device(), Device::CPU);
    EXPECT_EQ(t.storage_offset(), 0);
}

TEST(Phase1, MetadataOnShaped) {
    Tensor t({3, 4, 5}, 0.0f);
    EXPECT_EQ(t.dim(), 3);
    EXPECT_EQ(t.numel(), 60);
    EXPECT_EQ(t.sizes()[0], 3);
    EXPECT_EQ(t.sizes()[1], 4);
    EXPECT_EQ(t.sizes()[2], 5);
    EXPECT_EQ(t.strides()[0], 20);
    EXPECT_EQ(t.strides()[1], 5);
    EXPECT_EQ(t.strides()[2], 1);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_FALSE(t.is_empty());
}

TEST(Phase1, EmptyTensor) {
    Tensor t({0, 3});
    EXPECT_TRUE(t.is_empty());
    EXPECT_EQ(t.numel(), 0);
}

TEST(Phase1, StorageAccess) {
    Tensor t({2, 3}, 7.0f);
    EXPECT_NE(t.data_ptr(), nullptr);
    EXPECT_TRUE(t.storage().is_valid());
    EXPECT_NEAR(t.data_ptr()[0], 7.0f, EPS);
}

// ════════════════════════════════════════════════════════════════
// Phase 2: Constructors & Static Creators
// ════════════════════════════════════════════════════════════════

TEST(Phase2, ConstructWithValue) {
    Tensor t({2, 3}, 5.0f);
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(t.data_ptr()[i], 5.0f, EPS);
}

TEST(Phase2, Zeros) {
    Tensor t = Tensor::zeros({3, 3});
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(t.data_ptr()[i], 0.0f, EPS);
}

TEST(Phase2, Ones) {
    Tensor t = Tensor::ones({2, 2});
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(t.data_ptr()[i], 1.0f, EPS);
}

TEST(Phase2, Full) {
    Tensor t = Tensor::full({2, 2}, 3.14f);
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(t.data_ptr()[i], 3.14f, EPS);
}

TEST(Phase2, Arange) {
    Tensor t = Tensor::arange(0, 5, 1);
    EXPECT_EQ(t.numel(), 5);
    for (int i = 0; i < 5; ++i)
        EXPECT_NEAR(t.data_ptr()[i], static_cast<float>(i), EPS);
}

TEST(Phase2, ArangeStep) {
    Tensor t = Tensor::arange(0, 10, 2.5f);
    EXPECT_EQ(t.numel(), 4);
    EXPECT_NEAR(t.data_ptr()[0], 0.0f, EPS);
    EXPECT_NEAR(t.data_ptr()[1], 2.5f, EPS);
    EXPECT_NEAR(t.data_ptr()[2], 5.0f, EPS);
    EXPECT_NEAR(t.data_ptr()[3], 7.5f, EPS);
}

TEST(Phase2, Eye) {
    Tensor t = Tensor::eye(3);
    EXPECT_EQ(t.sizes()[0], 3);
    EXPECT_EQ(t.sizes()[1], 3);
    EXPECT_NEAR(t.at({0, 0}), 1.0f, EPS);
    EXPECT_NEAR(t.at({1, 1}), 1.0f, EPS);
    EXPECT_NEAR(t.at({2, 2}), 1.0f, EPS);
    EXPECT_NEAR(t.at({0, 1}), 0.0f, EPS);
}

TEST(Phase2, Diag) {
    Tensor v = Tensor::arange(1, 4, 1);
    Tensor d = Tensor::diag(v);
    EXPECT_NEAR(d.at({0, 0}), 1.0f, EPS);
    EXPECT_NEAR(d.at({1, 1}), 2.0f, EPS);
    EXPECT_NEAR(d.at({2, 2}), 3.0f, EPS);
    EXPECT_NEAR(d.at({0, 1}), 0.0f, EPS);
}

TEST(Phase2, Clone) {
    Tensor a({2, 2}, 3.0f);
    Tensor b = a.clone();
    b.set({0, 0}, 99.0f);
    EXPECT_NEAR(a.at({0, 0}), 3.0f, EPS);
    EXPECT_NEAR(b.at({0, 0}), 99.0f, EPS);
}

TEST(Phase2, FromBlob) {
    float data[] = {1, 2, 3, 4, 5, 6};
    Tensor t = Tensor::from_blob(data, {2, 3});
    EXPECT_NEAR(t.at({0, 0}), 1.0f, EPS);
    EXPECT_NEAR(t.at({1, 2}), 6.0f, EPS);
    data[0] = 99.0f;
    EXPECT_NEAR(t.at({0, 0}), 99.0f, EPS);
}

// ════════════════════════════════════════════════════════════════
// Phase 3: Indexing / Access
// ════════════════════════════════════════════════════════════════

TEST(Phase3, OperatorBracket) {
    Tensor t = Tensor::arange(0, 12, 1).view({3, 4});
    Tensor row = t[1];
    EXPECT_EQ(row.dim(), 1);
    EXPECT_EQ(row.sizes()[0], 4);
    EXPECT_NEAR(row.at({0}), 4.0f, EPS);
    EXPECT_NEAR(row.at({3}), 7.0f, EPS);
}

TEST(Phase3, AtAndSet) {
    Tensor t({3, 3}, 0.0f);
    t.set({1, 2}, 42.0f);
    EXPECT_NEAR(t.at({1, 2}), 42.0f, EPS);
    EXPECT_NEAR(t.at({0, 0}), 0.0f, EPS);
}

TEST(Phase3, Item) {
    Tensor t({1}, 7.5f);
    EXPECT_NEAR(t.item(), 7.5f, EPS);
}

TEST(Phase3, ItemThrows) {
    Tensor t({2, 2}, 1.0f);
    EXPECT_THROW(t.item(), std::runtime_error);
}

// ════════════════════════════════════════════════════════════════
// Phase 4: View / Shape Ops
// ════════════════════════════════════════════════════════════════

TEST(Phase4, View) {
    Tensor t = Tensor::arange(0, 12, 1);
    Tensor v = t.view({3, 4});
    EXPECT_EQ(v.dim(), 2);
    EXPECT_NEAR(v.at({2, 3}), 11.0f, EPS);
}

TEST(Phase4, ViewInferred) {
    Tensor t = Tensor::arange(0, 12, 1);
    Tensor v = t.view({3, -1});
    EXPECT_EQ(v.sizes()[1], 4);
}

TEST(Phase4, Reshape) {
    Tensor t = Tensor::arange(0, 6, 1).view({2, 3});
    Tensor r = t.reshape({3, 2});
    EXPECT_EQ(r.sizes()[0], 3);
    EXPECT_EQ(r.sizes()[1], 2);
}

TEST(Phase4, Flatten) {
    Tensor t({2, 3, 4}, 1.0f);
    Tensor f = t.flatten();
    EXPECT_EQ(f.dim(), 1);
    EXPECT_EQ(f.numel(), 24);
}

TEST(Phase4, Transpose) {
    Tensor t = Tensor::arange(0, 6, 1).view({2, 3});
    Tensor tr = t.transpose(0, 1);
    EXPECT_EQ(tr.sizes()[0], 3);
    EXPECT_EQ(tr.sizes()[1], 2);
    EXPECT_NEAR(tr.at({0, 1}), 3.0f, EPS);
    EXPECT_NEAR(tr.at({2, 0}), 2.0f, EPS);
}

TEST(Phase4, Permute) {
    Tensor t({2, 3, 4}, 1.0f);
    Tensor p = t.permute({2, 0, 1});
    EXPECT_EQ(p.sizes()[0], 4);
    EXPECT_EQ(p.sizes()[1], 2);
    EXPECT_EQ(p.sizes()[2], 3);
}

TEST(Phase4, Squeeze) {
    Tensor t({1, 3, 1, 4}, 1.0f);
    Tensor s = t.squeeze();
    EXPECT_EQ(s.dim(), 2);
    EXPECT_EQ(s.sizes()[0], 3);
    EXPECT_EQ(s.sizes()[1], 4);
}

TEST(Phase4, SqueezeDim) {
    Tensor t({1, 3, 1, 4}, 1.0f);
    Tensor s = t.squeeze(0);
    EXPECT_EQ(s.dim(), 3);
    EXPECT_EQ(s.sizes()[0], 3);
}

TEST(Phase4, Unsqueeze) {
    Tensor t({3, 4}, 1.0f);
    Tensor u = t.unsqueeze(0);
    EXPECT_EQ(u.dim(), 3);
    EXPECT_EQ(u.sizes()[0], 1);
    EXPECT_EQ(u.sizes()[1], 3);
    EXPECT_EQ(u.sizes()[2], 4);
}

TEST(Phase4, Expand) {
    Tensor t({1, 3}, 1.0f);
    t.set({0, 0}, 1.0f);
    t.set({0, 1}, 2.0f);
    t.set({0, 2}, 3.0f);
    Tensor e = t.expand({4, 3});
    EXPECT_EQ(e.sizes()[0], 4);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(e.at({i, 0}), 1.0f, EPS);
        EXPECT_NEAR(e.at({i, 1}), 2.0f, EPS);
        EXPECT_NEAR(e.at({i, 2}), 3.0f, EPS);
    }
}

TEST(Phase4, Narrow) {
    Tensor t = Tensor::arange(0, 10, 1);
    Tensor n = t.narrow(0, 2, 3);
    EXPECT_EQ(n.numel(), 3);
    EXPECT_NEAR(n.at({0}), 2.0f, EPS);
    EXPECT_NEAR(n.at({2}), 4.0f, EPS);
}

TEST(Phase4, Slice) {
    Tensor t = Tensor::arange(0, 10, 1);
    Tensor s = t.slice(0, 1, 8, 2);
    EXPECT_EQ(s.numel(), 4);
    EXPECT_NEAR(s.at({0}), 1.0f, EPS);
    EXPECT_NEAR(s.at({1}), 3.0f, EPS);
    EXPECT_NEAR(s.at({2}), 5.0f, EPS);
    EXPECT_NEAR(s.at({3}), 7.0f, EPS);
}

TEST(Phase4, Select) {
    Tensor t = Tensor::arange(0, 12, 1).view({3, 4});
    Tensor s = t.select(0, 1);
    EXPECT_EQ(s.dim(), 1);
    EXPECT_NEAR(s.at({0}), 4.0f, EPS);
}

TEST(Phase4, Contiguous) {
    Tensor t = Tensor::arange(0, 6, 1).view({2, 3});
    Tensor tr = t.transpose(0, 1);
    EXPECT_FALSE(tr.is_contiguous());
    Tensor c = tr.contiguous();
    EXPECT_TRUE(c.is_contiguous());
    EXPECT_NEAR(c.at({0, 0}), 0.0f, EPS);
    EXPECT_NEAR(c.at({0, 1}), 3.0f, EPS);
}

// ════════════════════════════════════════════════════════════════
// Phase 5: Copy / Ownership
// ════════════════════════════════════════════════════════════════

TEST(Phase5, CopyInplace) {
    Tensor a({2, 2}, 1.0f);
    Tensor b({2, 2}, 9.0f);
    a.copy_(b);
    EXPECT_NEAR(a.at({0, 0}), 9.0f, EPS);
}

TEST(Phase5, Detach) {
    Tensor a({2, 2}, 1.0f);
    a.set_requires_grad(true);
    Tensor b = a.detach();
    EXPECT_FALSE(b.requires_grad());
    b.set({0, 0}, 99.0f);
    EXPECT_NEAR(a.at({0, 0}), 99.0f, EPS);
}

TEST(Phase5, SharedStorage) {
    Tensor a({2, 2}, 1.0f);
    Tensor b = a[0];
    EXPECT_TRUE(a.is_shared_storage());
    EXPECT_TRUE(b.is_shared_storage());
    Tensor c = a.clone();
    EXPECT_FALSE(c.is_shared_storage());
}

// ════════════════════════════════════════════════════════════════
// Phase 6: Elementwise Ops
// ════════════════════════════════════════════════════════════════

TEST(Phase6, AddSubMulDiv) {
    Tensor a = Tensor::arange(1, 5, 1);
    Tensor b = Tensor::full({4}, 2.0f);
    Tensor sum = a.add(b);
    Tensor diff = a.sub(b);
    Tensor prod = a.mul(b);
    Tensor quot = a.div(b);
    EXPECT_NEAR(sum.at({0}), 3.0f, EPS);
    EXPECT_NEAR(diff.at({0}), -1.0f, EPS);
    EXPECT_NEAR(prod.at({0}), 2.0f, EPS);
    EXPECT_NEAR(quot.at({0}), 0.5f, EPS);
}

TEST(Phase6, Pow) {
    Tensor a = Tensor::full({3}, 2.0f);
    Tensor b = Tensor::full({3}, 3.0f);
    Tensor r = a.pow(b);
    EXPECT_NEAR(r.at({0}), 8.0f, EPS);
}

TEST(Phase6, UnaryOps) {
    Tensor t({1}, 2.0f);
    EXPECT_NEAR(t.neg().item(), -2.0f, EPS);
    EXPECT_NEAR(t.exp().item(), std::exp(2.0f), EPS);
    EXPECT_NEAR(t.log().item(), std::log(2.0f), EPS);
    EXPECT_NEAR(t.sqrt().item(), std::sqrt(2.0f), EPS);
    Tensor n({1}, -3.0f);
    EXPECT_NEAR(n.abs().item(), 3.0f, EPS);
    EXPECT_NEAR(n.relu().item(), 0.0f, EPS);
}

TEST(Phase6, Activations) {
    Tensor t({1}, 0.0f);
    EXPECT_NEAR(t.sigmoid().item(), 0.5f, EPS);
    EXPECT_NEAR(t.tanh().item(), 0.0f, EPS);
}

TEST(Phase6, ScalarOps) {
    Tensor t = Tensor::arange(1, 4, 1);
    Tensor r = t.add_scalar(10.0f);
    EXPECT_NEAR(r.at({0}), 11.0f, EPS);
    EXPECT_NEAR(r.at({2}), 13.0f, EPS);
    r = t.mul_scalar(3.0f);
    EXPECT_NEAR(r.at({0}), 3.0f, EPS);
}

TEST(Phase6, InplaceOps) {
    Tensor a = Tensor::ones({3});
    Tensor b = Tensor::full({3}, 2.0f);
    a.add_(b);
    EXPECT_NEAR(a.at({0}), 3.0f, EPS);
    a.mul_(b);
    EXPECT_NEAR(a.at({0}), 6.0f, EPS);
}

TEST(Phase6, Broadcasting) {
    Tensor a = Tensor::arange(0, 6, 1).view({2, 3});
    Tensor b = Tensor::arange(0, 3, 1);
    Tensor r = a.add(b);
    EXPECT_NEAR(r.at({0, 0}), 0.0f, EPS);
    EXPECT_NEAR(r.at({0, 2}), 4.0f, EPS);
    EXPECT_NEAR(r.at({1, 0}), 3.0f, EPS);
    EXPECT_NEAR(r.at({1, 2}), 7.0f, EPS);
}

// ════════════════════════════════════════════════════════════════
// Phase 7: Reduction Ops
// ════════════════════════════════════════════════════════════════

TEST(Phase7, SumAll) {
    Tensor t = Tensor::arange(1, 5, 1);
    Tensor s = t.sum();
    EXPECT_NEAR(s.item(), 10.0f, EPS);
}

TEST(Phase7, SumDim) {
    Tensor t = Tensor::arange(0, 6, 1).view({2, 3});
    Tensor s = t.sum(0);
    EXPECT_EQ(s.dim(), 1);
    EXPECT_NEAR(s.at({0}), 3.0f, EPS);
    EXPECT_NEAR(s.at({1}), 5.0f, EPS);
    EXPECT_NEAR(s.at({2}), 7.0f, EPS);
}

TEST(Phase7, Mean) {
    Tensor t = Tensor::arange(1, 5, 1);
    Tensor m = t.mean();
    EXPECT_NEAR(m.item(), 2.5f, EPS);
}

TEST(Phase7, MaxMin) {
    Tensor t = Tensor::arange(0, 6, 1).view({2, 3});
    EXPECT_NEAR(t.max().item(), 5.0f, EPS);
    EXPECT_NEAR(t.min().item(), 0.0f, EPS);
}

TEST(Phase7, ArgmaxArgmin) {
    float data[] = {3, 1, 4, 1, 5, 9};
    Tensor t = Tensor::from_blob(data, {6});
    EXPECT_NEAR(t.argmax().item(), 5.0f, EPS);
    EXPECT_NEAR(t.argmin().item(), 1.0f, EPS);
}

// ════════════════════════════════════════════════════════════════
// Phase 8: Comparison Ops
// ════════════════════════════════════════════════════════════════

TEST(Phase8, Comparisons) {
    Tensor a = Tensor::arange(0, 5, 1);
    Tensor b = Tensor::full({5}, 2.0f);
    Tensor eq = a.eq(b);
    Tensor lt = a.lt(b);
    Tensor gt = a.gt(b);
    EXPECT_NEAR(eq.at({2}), 1.0f, EPS);
    EXPECT_NEAR(eq.at({0}), 0.0f, EPS);
    EXPECT_NEAR(lt.at({0}), 1.0f, EPS);
    EXPECT_NEAR(lt.at({3}), 0.0f, EPS);
    EXPECT_NEAR(gt.at({3}), 1.0f, EPS);
    EXPECT_NEAR(gt.at({1}), 0.0f, EPS);
}

// ════════════════════════════════════════════════════════════════
// Phase 8: Linear Algebra
// ════════════════════════════════════════════════════════════════

TEST(Phase8, MM) {
    Tensor a = Tensor::eye(3);
    Tensor b = Tensor::arange(0, 9, 1).view({3, 3});
    Tensor r = a.mm(b);
    for (int i = 0; i < 9; ++i)
        EXPECT_NEAR(r.data_ptr()[i], b.data_ptr()[i], EPS);
}

TEST(Phase8, MatmulVec) {
    Tensor a = Tensor::eye(3);
    Tensor b = Tensor::arange(1, 4, 1);
    Tensor r = a.matmul(b);
    EXPECT_NEAR(r.at({0}), 1.0f, EPS);
    EXPECT_NEAR(r.at({1}), 2.0f, EPS);
    EXPECT_NEAR(r.at({2}), 3.0f, EPS);
}

TEST(Phase8, Dot) {
    Tensor a = Tensor::arange(1, 4, 1);
    Tensor b = Tensor::arange(1, 4, 1);
    EXPECT_NEAR(a.dot(b).item(), 14.0f, EPS);
}

TEST(Phase8, Outer) {
    Tensor a = Tensor::arange(1, 4, 1);
    Tensor b = Tensor::arange(1, 3, 1);
    Tensor r = a.outer(b);
    EXPECT_EQ(r.sizes()[0], 3);
    EXPECT_EQ(r.sizes()[1], 2);
    EXPECT_NEAR(r.at({0, 0}), 1.0f, EPS);
    EXPECT_NEAR(r.at({2, 1}), 6.0f, EPS);
}

TEST(Phase8, BMM) {
    Tensor a = Tensor::ones({2, 3, 4});
    Tensor b = Tensor::ones({2, 4, 5});
    Tensor r = a.bmm(b);
    EXPECT_EQ(r.sizes()[0], 2);
    EXPECT_EQ(r.sizes()[1], 3);
    EXPECT_EQ(r.sizes()[2], 5);
    EXPECT_NEAR(r.at({0, 0, 0}), 4.0f, EPS);
}

// ════════════════════════════════════════════════════════════════
// Phase 8: Concatenation / Splitting
// ════════════════════════════════════════════════════════════════

TEST(Phase8, Cat) {
    Tensor a = Tensor::ones({2, 3});
    Tensor b = Tensor::zeros({2, 3});
    Tensor c = Tensor::cat({a, b}, 0);
    EXPECT_EQ(c.sizes()[0], 4);
    EXPECT_EQ(c.sizes()[1], 3);
    EXPECT_NEAR(c.at({0, 0}), 1.0f, EPS);
    EXPECT_NEAR(c.at({2, 0}), 0.0f, EPS);
}

TEST(Phase8, Stack) {
    Tensor a = Tensor::ones({3});
    Tensor b = Tensor::zeros({3});
    Tensor c = Tensor::stack({a, b}, 0);
    EXPECT_EQ(c.sizes()[0], 2);
    EXPECT_EQ(c.sizes()[1], 3);
}

TEST(Phase8, Split) {
    Tensor t = Tensor::arange(0, 10, 1);
    auto parts = t.split(3);
    EXPECT_EQ(parts.size(), 4u);
    EXPECT_EQ(parts[0].numel(), 3);
    EXPECT_EQ(parts[3].numel(), 1);
}

TEST(Phase8, Chunk) {
    Tensor t = Tensor::arange(0, 10, 1);
    auto parts = t.chunk(3);
    EXPECT_EQ(parts.size(), 3u);
}

// ════════════════════════════════════════════════════════════════
// Phase 9: Autograd
// ════════════════════════════════════════════════════════════════

TEST(Phase9, RequiresGrad) {
    Tensor t({2, 2}, 1.0f);
    EXPECT_FALSE(t.requires_grad());
    t.set_requires_grad(true);
    EXPECT_TRUE(t.requires_grad());
}

TEST(Phase9, GradSetGet) {
    Tensor t({2, 2}, 1.0f);
    t.set_requires_grad(true);
    Tensor g({2, 2}, 0.5f);
    t.set_grad(g);
    EXPECT_NEAR(t.grad().at({0, 0}), 0.5f, EPS);
}

TEST(Phase9, Backward) {
    Tensor t({2, 2}, 1.0f);
    t.set_requires_grad(true);
    Tensor grad_out({2, 2}, 2.0f);
    t.backward(grad_out);
    EXPECT_NEAR(t.grad().at({0, 0}), 2.0f, EPS);
}

TEST(Phase9, BackwardAccumulates) {
    Tensor t({2}, 1.0f);
    t.set_requires_grad(true);
    t.backward(Tensor({2}, 1.0f));
    t.backward(Tensor({2}, 3.0f));
    EXPECT_NEAR(t.grad().at({0}), 4.0f, EPS);
}

TEST(Phase9, ZeroGrad) {
    Tensor t({2}, 1.0f);
    t.set_requires_grad(true);
    t.backward(Tensor({2}, 5.0f));
    t.zero_grad();
    Tensor g = t.grad();
    EXPECT_EQ(g.dim(), 0);
}

TEST(Phase9, DetachSeversGrad) {
    Tensor t({2}, 1.0f);
    t.set_requires_grad(true);
    Tensor d = t.detach();
    EXPECT_FALSE(d.requires_grad());
}

// ════════════════════════════════════════════════════════════════
// Phase 10: Utility / Debug
// ════════════════════════════════════════════════════════════════

TEST(Phase10, ShapeString) {
    Tensor t({2, 3, 4}, 0.0f);
    EXPECT_EQ(t.shape_string(), "(2, 3, 4)");
}

TEST(Phase10, StrideString) {
    Tensor t({2, 3, 4}, 0.0f);
    EXPECT_EQ(t.stride_string(), "(12, 4, 1)");
}

TEST(Phase10, ToString) {
    Tensor t({2}, 1.0f);
    std::string s = t.to_string();
    EXPECT_FALSE(s.empty());
    EXPECT_NE(s.find("Tensor"), std::string::npos);
}

TEST(Phase10, OstreamOperator) {
    Tensor t({2}, 1.0f);
    std::ostringstream ss;
    ss << t;
    EXPECT_FALSE(ss.str().empty());
}

TEST(Phase10, ToDevice) {
    Tensor t({2}, 1.0f);
    Tensor c = t.to(Device::CUDA);
    EXPECT_EQ(c.device(), Device::CUDA);
    EXPECT_NEAR(c.at({0}), 1.0f, EPS);
}

TEST(Phase10, AsType) {
    Tensor t({2}, 1.0f);
    Tensor c = t.astype(DType::Float64);
    EXPECT_EQ(c.dtype(), DType::Float64);
}

TEST(Phase10, PinMemory) {
    Tensor t({2}, 1.0f);
    Tensor p = t.pin_memory();
    EXPECT_NEAR(p.at({0}), 1.0f, EPS);
}

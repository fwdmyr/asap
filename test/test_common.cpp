#include "../include/common.hpp"
#include <gtest/gtest.h>

namespace {

TEST(Argsort, EmptyVector) {
  const auto v = std::vector<double>{};
  const auto expected_idx = std::vector<std::size_t>{};

  const auto idx = asap::argsort(v);

  EXPECT_EQ(idx, expected_idx);
}

TEST(Argsort, UniqueElementVector) {
  const auto v = std::vector<double>{3.0, 1.0, -1.0, 4.0, 9.0};
  const auto expected_idx = std::vector<std::size_t>{2, 1, 0, 3, 4};

  const auto idx = asap::argsort(v);

  EXPECT_EQ(idx, expected_idx);
}

TEST(Argsort, DuplicateElementsVector) {
  const auto v = std::vector<double>{3.0, 1.0, -1.0, 4.0, 1.0, 9.0};
  const auto expected_idx = std::vector<std::size_t>{2, 1, 4, 0, 3, 5};

  const auto idx = asap::argsort(v);

  EXPECT_EQ(idx, expected_idx);
}

class ReorderFixture : public ::testing::Test {
protected:
  std::vector<double> v_{1.0, 2.0, 3.0, 4.0, 5.0};
};

TEST_F(ReorderFixture, EmptyIndexVector) {
  const auto idx = std::vector<std::size_t>{};
  const auto expected_v = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0};

  asap::reorder(idx, v_);

  EXPECT_EQ(v_, expected_v);
}

TEST_F(ReorderFixture, SortedIndexVector) {
  const auto idx = std::vector<std::size_t>{0, 1, 2, 3, 4};
  const auto expected_v = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0};

  asap::reorder(idx, v_);

  EXPECT_EQ(v_, expected_v);
}

TEST_F(ReorderFixture, UnsortedPartialIndexVector) {
  const auto idx = std::vector<std::size_t>{2, 0, 1};
  const auto expected_v = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0};

  asap::reorder(idx, v_);

  EXPECT_EQ(v_, expected_v);
}

TEST_F(ReorderFixture, UnsortedIndexVector) {
  const auto idx = std::vector<std::size_t>{2, 0, 1, 4, 3};
  const auto expected_v = std::vector<double>{3.0, 1.0, 2.0, 5.0, 4.0};

  asap::reorder(idx, v_);

  EXPECT_EQ(v_, expected_v);
}

} // namespace

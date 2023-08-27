#include "../include/sparse_jonker_volgenant_solver.hpp"
#include <gtest/gtest.h>

namespace {

template <typename MatrixType>
class SparseJonkerVolgenantSolverFixture : public ::testing::Test {
public:
  using Type = MatrixType;
};

using MatrixTypes =
    ::testing::Types<Eigen::SparseMatrix<double, Eigen::RowMajor>,
                     Eigen::SparseMatrix<double, Eigen::ColMajor>>;
TYPED_TEST_SUITE(SparseJonkerVolgenantSolverFixture, MatrixTypes);

TYPED_TEST(SparseJonkerVolgenantSolverFixture, Solve_EqualWeightSquareMatrix) {
  using SparseMatrixT = typename TestFixture::Type;

  auto sm = SparseMatrixT(3U, 3U);
  sm.insert(0U, 0U) = 1.0;
  sm.insert(0U, 1U) = 1.0;
  sm.insert(0U, 2U) = 1.0;
  sm.insert(1U, 0U) = 1.0;
  sm.insert(2U, 1U) = 1.0;
  const auto expected_col_idx = std::vector<Eigen::Index>{2, 0, 1};

  const auto res = asap::SparseJonkerVolgenantSolver::Solve(std::move(sm));

  EXPECT_EQ(res.col_idx, expected_col_idx);
}

TYPED_TEST(SparseJonkerVolgenantSolverFixture, Solve_DenseSquareMatrix) {
  using SparseMatrixT = typename TestFixture::Type;

  auto sm = SparseMatrixT(3U, 3U);
  sm.insert(0U, 0U) = 3.0;
  sm.insert(0U, 1U) = 3.0;
  sm.insert(0U, 2U) = 6.0;
  sm.insert(1U, 0U) = 4.0;
  sm.insert(1U, 1U) = 3.0;
  sm.insert(1U, 2U) = 5.0;
  sm.insert(2U, 0U) = 10.0;
  sm.insert(2U, 1U) = 1.0;
  sm.insert(2U, 2U) = 8.0;
  const auto expected_col_idx = std::vector<Eigen::Index>{0, 2, 1};

  const auto res = asap::SparseJonkerVolgenantSolver::Solve(std::move(sm));

  EXPECT_EQ(res.col_idx, expected_col_idx);
}

TYPED_TEST(SparseJonkerVolgenantSolverFixture,
           Solve_SparseWideRectangularMatrix) {
  using SparseMatrixT = typename TestFixture::Type;

  auto sm = SparseMatrixT(2U, 3U);
  sm.insert(0U, 1U) = 1.0;
  sm.insert(0U, 2U) = 1.0;
  sm.insert(1U, 1U) = 2.0;
  sm.insert(1U, 2U) = 3.0;
  const auto expected_row_idx = std::vector<Eigen::Index>{0, 1};
  const auto expected_col_idx = std::vector<Eigen::Index>{2, 1};

  const auto res = asap::SparseJonkerVolgenantSolver::Solve(std::move(sm));

  EXPECT_EQ(res.row_idx, expected_row_idx);
  EXPECT_EQ(res.col_idx, expected_col_idx);
}

TYPED_TEST(SparseJonkerVolgenantSolverFixture,
           Solve_SparseTallRectangularMatrix) {
  using SparseMatrixT = typename TestFixture::Type;

  auto sm = SparseMatrixT(3U, 2U);
  sm.insert(0U, 1U) = 1.0;
  sm.insert(1U, 0U) = 3.0;
  sm.insert(1U, 1U) = 1.0;
  sm.insert(2U, 0U) = 1.0;
  sm.insert(2U, 1U) = 4.0;
  const auto expected_row_idx = std::vector<Eigen::Index>{0, 2};
  const auto expected_col_idx = std::vector<Eigen::Index>{1, 0};

  const auto res = asap::SparseJonkerVolgenantSolver::Solve(std::move(sm));

  EXPECT_EQ(res.row_idx, expected_row_idx);
  EXPECT_EQ(res.col_idx, expected_col_idx);
}

} // namespace

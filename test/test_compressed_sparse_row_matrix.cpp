#include "../include/compressed_sparse_row_matrix.hpp"
#include <gtest/gtest.h>

namespace {

TEST(TestCompressedSparseRowMatrix, ScipySquareExample) {
  auto mat = Eigen::SparseMatrix<double, Eigen::RowMajor>(3U, 3U);
  mat.insert(0U, 0U) = 1.0;
  mat.insert(0U, 2U) = 2.0;
  mat.insert(1U, 2U) = 3.0;
  mat.insert(2U, 0U) = 4.0;
  mat.insert(2U, 1U) = 5.0;
  mat.insert(2U, 2U) = 6.0;

  const auto expected_indptr = std::vector<Eigen::Index>{0, 2, 3, 6};
  const auto expected_indices = std::vector<Eigen::Index>{0, 2, 2, 0, 1, 2};
  const auto expected_data = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  const auto csr =
      asap::CompressedSparseRowMatrix<double>{std::move(mat)};

  EXPECT_EQ(csr.row_ptr, expected_indptr);
  EXPECT_EQ(csr.col_ind, expected_indices);
  EXPECT_EQ(csr.val, expected_data);
}

TEST(TestCompressedSparseRowMatrix, ScipyWideRectangularExample) {
  auto mat = Eigen::SparseMatrix<double, Eigen::RowMajor>(2U, 3U);
  mat.insert(0U, 0U) = 1.0;
  mat.insert(0U, 1U) = 2.0;
  mat.insert(1U, 2U) = 3.0;

  const auto expected_indptr = std::vector<Eigen::Index>{0, 2, 3};
  const auto expected_indices = std::vector<Eigen::Index>{0, 1, 2};
  const auto expected_data = std::vector<double>{1.0, 2.0, 3.0};

  const auto csr =
      asap::CompressedSparseRowMatrix<double>{std::move(mat)};

  EXPECT_EQ(csr.row_ptr, expected_indptr);
  EXPECT_EQ(csr.col_ind, expected_indices);
  EXPECT_EQ(csr.val, expected_data);
}

TEST(TestCompressedSparseRowMatrix, ScipyTallRectangularExample) {
  auto mat = Eigen::SparseMatrix<double, Eigen::RowMajor>(3U, 2U);
  mat.insert(0U, 0U) = 1.0;
  mat.insert(0U, 1U) = 2.0;
  mat.insert(1U, 0U) = 3.0;
  mat.insert(2U, 1U) = 4.0;

  const auto expected_indptr = std::vector<Eigen::Index>{0, 2, 3, 4};
  const auto expected_indices = std::vector<Eigen::Index>{0, 1, 0, 1};
  const auto expected_data = std::vector<double>{1.0, 2.0, 3.0, 4.0};

  const auto csr =
      asap::CompressedSparseRowMatrix<double>{std::move(mat)};

  EXPECT_EQ(csr.row_ptr, expected_indptr);
  EXPECT_EQ(csr.col_ind, expected_indices);
  EXPECT_EQ(csr.val, expected_data);
}

TEST(TestCompressedSparseRowMatrix, WikipediaExample) {
  auto mat = Eigen::SparseMatrix<double, Eigen::RowMajor>(4U, 5U);
  mat.insert(0U, 0U) = 10.0;
  mat.insert(0U, 3U) = 12.0;
  mat.insert(1U, 2U) = 11.0;
  mat.insert(1U, 4U) = 13.0;
  mat.insert(2U, 1U) = 16.0;
  mat.insert(3U, 2U) = 11.0;
  mat.insert(3U, 4U) = 13.0;

  const auto expected_row_ptr = std::vector<Eigen::Index>{0, 2, 4, 5, 7};
  const auto expected_col_ind = std::vector<Eigen::Index>{0, 3, 2, 4, 1, 2, 4};
  const auto expected_val =
      std::vector<double>{10.0, 12.0, 11.0, 13.0, 16.0, 11.0, 13.0};

  const auto csr =
      asap::CompressedSparseRowMatrix<double>{std::move(mat)};

  EXPECT_EQ(csr.row_ptr, expected_row_ptr);
  EXPECT_EQ(csr.col_ind, expected_col_ind);
  EXPECT_EQ(csr.val, expected_val);
}

} // namespace

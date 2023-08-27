#include "../include/sparse_jonker_volgenant_solver.hpp"
#include <gtest/gtest.h>

namespace {

TEST(TestSuiteDummy, UnitTestDummy) {
  auto mat = Eigen::SparseMatrix<double, Eigen::RowMajor>(3U, 3U);

  mat.insert(0U, 0U) = 1;
  mat.insert(0U, 2U) = 2;
  mat.insert(1U, 2U) = 3;
  mat.insert(2U, 0U) = 4;
  mat.insert(2U, 1U) = 5;
  mat.insert(2U, 2U) = 6;

  const auto csr = asap::CompressedSparseRowRepresentation<double>{mat};

  const auto ret = asap::SparseJonkerVolgenantSolver::Solve(csr);

  for (std::size_t i = 0; i < ret.first.size(); ++i) {
      std::cerr << ret.first[i] << ' ';
  }
  std::cerr << '\n';
  for (std::size_t i = 0; i < ret.second.size(); ++i) {
      std::cerr << ret.second[i] << ' ';
  }
  std::cerr << '\n';
}

} // namespace

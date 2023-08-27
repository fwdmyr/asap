#ifndef ASAP_SPARSE_JONKER_VOLGENANT_SOLVER_HPP
#define ASAP_SPARSE_JONKER_VOLGENANT_SOLVER_HPP

#include "common.hpp"
#include "sparse_jonker_volgenant_solver_impl.hpp"

namespace asap {

struct Result {
  std::vector<Eigen::Index> row_idx{};
  std::vector<Eigen::Index> col_idx{};
  bool valid{};
};

class SparseJonkerVolgenantSolver {
public:
  template <typename SparseMatrixT>
  [[nodiscard]] static std::enable_if_t<is_col_major<SparseMatrixT>, Result>
  Solve(SparseMatrixT &&sm);

  template <typename SparseMatrixT>
  [[nodiscard]] static std::enable_if_t<is_row_major<SparseMatrixT>, Result>
  Solve(SparseMatrixT &&sm);
};

template <typename SparseMatrixT>
std::enable_if_t<is_col_major<SparseMatrixT>, Result>
SparseJonkerVolgenantSolver::Solve(SparseMatrixT &&sm) {
  auto sm_row_major = static_cast<
      Eigen::SparseMatrix<typename SparseMatrixT::Scalar, Eigen::RowMajor>>(sm);
  return SparseJonkerVolgenantSolver::Solve(std::move(sm_row_major));
}

template <typename SparseMatrixT>
std::enable_if_t<is_row_major<SparseMatrixT>, Result>
SparseJonkerVolgenantSolver::Solve(SparseMatrixT &&sm) {
  using ScalarT = typename SparseMatrixT::Scalar;

  const auto transpose = sm.rows() > sm.cols();

  auto csr = (transpose)
                 ? CompressedSparseRowRepresentation<ScalarT>{sm.transpose()}
                 : CompressedSparseRowRepresentation<ScalarT>{sm};

  auto a = std::vector<Eigen::Index>(std::min(csr.rows(), csr.cols()));
  std::iota(a.begin(), a.end(), 0);

  auto valid = bool{};
  auto b = internal::lapjvsp(csr.row_ptr(), csr.col_ind(), csr.val(),
                             csr.rows(), csr.cols(), valid);

  if (transpose) {
    const auto idx = internal::argsort(b);
    internal::reorder(idx, a);
    internal::reorder(idx, b);
  }

  return (transpose) ? Result{std::move(b), std::move(a), valid}
                     : Result{std::move(a), std::move(b), valid};
}

} // namespace asap

#endif

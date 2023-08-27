#include "sparse_jonker_volgenant_solver_impl.hpp"
#include <eigen3/Eigen/src/Core/util/Constants.h>

namespace asap {

template <typename T>
static constexpr auto is_row_major =
    std::is_same_v<std::decay_t<T>,
                   Eigen::SparseMatrix<typename T::Scalar, Eigen::RowMajor>>;
template <typename T>
static constexpr auto is_col_major =
    std::is_same_v<std::decay_t<T>,
                   Eigen::SparseMatrix<typename T::Scalar, Eigen::ColMajor>>;

class SparseJonkerVolgenantSolver {
public:
  using Result =
      std::pair<std::vector<Eigen::Index>, std::vector<Eigen::Index>>;

  template <typename SparseMatrixT>
  static std::enable_if_t<is_col_major<SparseMatrixT>, Result>
  Solve(SparseMatrixT &&sm);

  template <typename SparseMatrixT>
  static std::enable_if_t<is_row_major<SparseMatrixT>, Result>
  Solve(SparseMatrixT &&sm);
};

template <typename SparseMatrixT>
std::enable_if_t<is_col_major<SparseMatrixT>,
                 SparseJonkerVolgenantSolver::Result>
SparseJonkerVolgenantSolver::Solve(SparseMatrixT &&sm) {
  auto sm_row_major = static_cast<
      Eigen::SparseMatrix<typename SparseMatrixT::Scalar, Eigen::RowMajor>>(sm);
  return SparseJonkerVolgenantSolver::Solve(std::move(sm_row_major));
}

template <typename SparseMatrixT>
std::enable_if_t<is_row_major<SparseMatrixT>,
                 SparseJonkerVolgenantSolver::Result>
SparseJonkerVolgenantSolver::Solve(SparseMatrixT &&sm) {

  auto csr =
      CompressedSparseRowRepresentation<typename SparseMatrixT::Scalar>{sm};

  const auto transpose = sm.rows() > sm.cols();

  if (transpose) {
    csr = CompressedSparseRowRepresentation<typename SparseMatrixT::Scalar>{
        sm.transpose()};
  }

  auto a = std::vector<Eigen::Index>{};
  std::generate_n(std::back_inserter(a), std::min(csr.rows(), csr.cols()),
                  [incr = 0]() mutable { return incr++; });

  auto b = internal::lapjvsp(csr.row_ptr(), csr.col_ind(), csr.val(),
                             csr.rows(), csr.cols());

  return std::make_pair(std::move(a), std::move(b));
}

} // namespace asap

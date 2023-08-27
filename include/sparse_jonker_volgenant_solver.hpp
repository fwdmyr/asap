#include "sparse_jonker_volgenant_solver_impl.hpp"
#include <algorithm>
#include <numeric>

namespace asap {

template <template <typename, typename> typename Container, typename T,
          typename Alloc = std::allocator<T>>
[[nodiscard]] auto argsort(const Container<T, Alloc> &c) {

  auto idx = Container<std::size_t, std::allocator<std::size_t>>(c.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::stable_sort(idx.begin(), idx.end(),
                   [&c](auto lhs, auto rhs) { return c[lhs] < c[rhs]; });

  return idx;
}

template <typename OrderIter, typename ValueIter>
void reorder(OrderIter order_begin, OrderIter order_end, ValueIter v) {
  using IndexT = typename std::iterator_traits<OrderIter>::value_type;

  auto remaining = order_end - 1 - order_begin;
  for (IndexT s = IndexT{}, d; remaining > 0; ++s) {
    for (d = order_begin[s]; d > s; d = order_begin[d])
      if (d == s) {
        --remaining;
        auto temp = v[s];
        while (d = order_begin[d], d != s) {
          std::swap(temp, v[d]);
          --remaining;
        }
        v[s] = temp;
      }
  }
}

class SparseJonkerVolgenantSolver {
public:
  using Result =
      std::pair<std::vector<Eigen::Index>, std::vector<Eigen::Index>>;

  template <typename SparseMatrixT>
  [[nodiscard]] static std::enable_if_t<is_col_major<SparseMatrixT>, Result>
  Solve(SparseMatrixT &&sm);

  template <typename SparseMatrixT>
  [[nodiscard]] static std::enable_if_t<is_row_major<SparseMatrixT>, Result>
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

  auto a = std::vector<Eigen::Index>(std::min(csr.rows(), csr.cols()));
  std::iota(a.begin(), a.end(), 0);

  auto b = internal::lapjvsp(csr.row_ptr(), csr.col_ind(), csr.val(),
                             csr.rows(), csr.cols());

  if (transpose) {
    const auto idx = argsort(b);
    reorder(idx.cbegin(), idx.cend(), a.begin());
    reorder(idx.cbegin(), idx.cend(), b.begin());
  }

  return (transpose) ? std::make_pair(std::move(b), std::move(a))
                     : std::make_pair(std::move(a), std::move(b));
}

} // namespace asap

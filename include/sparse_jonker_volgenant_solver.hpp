#include "sparse_jonker_volgenant_solver_impl.hpp"

namespace asap {

class SparseJonkerVolgenantSolver {
public:
  template <typename SparseReprT> static auto Solve(SparseReprT &&sm);
};

template <typename SparseReprT>
auto SparseJonkerVolgenantSolver::Solve(SparseReprT &&sm) {
  auto b = internal::lapjvsp(sm.row_ptr(), sm.col_ind(), sm.val(), sm.rows(),
                             sm.cols());

  auto a = std::vector<int>{};

  std::generate_n(std::back_inserter(a), std::min(sm.rows(), sm.cols()),
                  [incr = 0]() mutable { return incr++; });

  return std::make_pair(std::move(a), std::move(b));
}

} // namespace asap

#ifndef ASAP_COMPRESSED_SPARSE_ROW_MATRIX_HPP
#define ASAP_COMPRESSED_SPARSE_ROW_MATRIX_HPP

#include "common.hpp"
#include <Eigen/SparseCore>
#include <cassert>

namespace asap {

template <typename T>
static constexpr auto is_row_major =
    std::is_same_v<std::decay_t<T>,
                   Eigen::SparseMatrix<typename T::Scalar, Eigen::RowMajor>>;
template <typename T>
static constexpr auto is_col_major =
    std::is_same_v<std::decay_t<T>,
                   Eigen::SparseMatrix<typename T::Scalar, Eigen::ColMajor>>;

template <typename T> struct CompressedSparseRowMatrix {
  explicit CompressedSparseRowMatrix(
      const Eigen::SparseMatrix<T, Eigen::RowMajor> &s) noexcept;

  std::vector<T> val{};
  std::vector<Eigen::Index> col_ind{};
  std::vector<Eigen::Index> row_ptr{};
  Eigen::Index rows{};
  Eigen::Index cols{};
};

template <typename T>
CompressedSparseRowMatrix<T>::CompressedSparseRowMatrix(
    const Eigen::SparseMatrix<T, Eigen::RowMajor> &sm) noexcept
    : val{sm.valuePtr(), sm.valuePtr() + sm.nonZeros()},
      col_ind{sm.innerIndexPtr(), sm.innerIndexPtr() + sm.nonZeros()},
      row_ptr{sm.outerIndexPtr(), sm.outerIndexPtr() + sm.outerSize()},
      rows{sm.rows()}, cols{sm.cols()} {
  row_ptr.push_back(val.size());
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const CompressedSparseRowMatrix<T> &csr) {
  os << "CSR Matrix Representation" << '\n';
  os << "Dimension ( " << csr.rows << " x " << csr.cols << " )" << '\n';
  os << "Val       ( ";
  os << csr.val << ')' << '\n';
  os << "ColInd    ( ";
  os << csr.col_ind << ')' << '\n';
  os << "RowPtr    ( ";
  os << csr.row_ptr << ')' << '\n';
  return os;
}

} // namespace asap

#endif

#ifndef ASAP_SPARSE_MATRIX_TRAITS_HPP
#define ASAP_SPARSE_MATRIX_TRAITS_HPP

#include <eigen3/Eigen/SparseCore>
#include <type_traits>

namespace asap {

template <typename T>
static constexpr auto is_row_major_v =
    std::is_same_v<std::decay_t<T>,
                   Eigen::SparseMatrix<typename T::Scalar, Eigen::RowMajor>>;
template <typename T>
static constexpr auto is_col_major_v =
    std::is_same_v<std::decay_t<T>,
                   Eigen::SparseMatrix<typename T::Scalar, Eigen::ColMajor>>;

} // namespace asap

#endif

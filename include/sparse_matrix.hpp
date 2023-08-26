#include <Eigen/SparseCore>
#include <vector>

namespace asap {

template <template <typename, typename> typename Container, typename T,
          typename Alloc>
std::ostream &operator<<(std::ostream &os, const Container<T, Alloc> &c) {
  for (const auto &e : c) {
    os << e << ' ';
  }
  return os;
}

template <typename T> class CompressedSparseRowRepresentation {
public:
  explicit CompressedSparseRowRepresentation(
      const Eigen::SparseMatrix<T, Eigen::RowMajor> &sm) noexcept;

  std::ostream &print(std::ostream &os) const noexcept;

  const auto &val() const noexcept;
  const auto &col_ind() const noexcept;
  const auto &row_ptr() const noexcept;
  auto rows() const noexcept;
  auto cols() const noexcept;

private:
  std::vector<T> m_val{};
  std::vector<std::size_t> m_col_ind{};
  std::vector<std::size_t> m_row_ptr{};
  Eigen::Index m_rows{};
  Eigen::Index m_cols{};
};

template <typename T>
CompressedSparseRowRepresentation<T>::CompressedSparseRowRepresentation(
    const Eigen::SparseMatrix<T, Eigen::RowMajor> &sm) noexcept
    : m_val{sm.valuePtr(), sm.valuePtr() + sm.nonZeros()},
      m_col_ind{sm.innerIndexPtr(), sm.innerIndexPtr() + sm.nonZeros()},
      m_row_ptr{sm.outerIndexPtr(), sm.outerIndexPtr() + sm.outerSize()},
      m_rows{sm.rows()}, m_cols{sm.cols()} {
  if (!m_row_ptr.empty()) {
    m_row_ptr.push_back(m_row_ptr.back() + sm.nonZeros() - sm.innerSize());
  }
}

template <typename T>
std::ostream &
CompressedSparseRowRepresentation<T>::print(std::ostream &os) const noexcept {
  os << "CSR Matrix Representation" << '\n';
  os << "Dimension ( " << m_rows << " x " << m_cols << " )" << '\n';
  os << "Val       ( ";
  os << m_val << ')' << '\n';
  os << "ColInd    ( ";
  os << m_col_ind << ')' << '\n';
  os << "RowPtr    ( ";
  os << m_row_ptr << ')' << '\n';
  return os;
}

template <typename T>
std::ostream &operator<<(std::ostream &os,
                         const CompressedSparseRowRepresentation<T> &csr) {
  return csr.print(os);
}

template <typename T>
const auto &CompressedSparseRowRepresentation<T>::val() const noexcept {
  return m_val;
}

template <typename T>
const auto &CompressedSparseRowRepresentation<T>::col_ind() const noexcept {
  return m_col_ind;
}

template <typename T>
const auto &CompressedSparseRowRepresentation<T>::row_ptr() const noexcept {
  return m_row_ptr;
}

template <typename T>
auto CompressedSparseRowRepresentation<T>::rows() const noexcept {
  return m_rows;
}

template <typename T>
auto CompressedSparseRowRepresentation<T>::cols() const noexcept {
  return m_cols;
}

} // namespace asap

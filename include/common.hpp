#ifndef ASAP_COMMON_HPP
#define ASAP_COMMON_HPP

#include <algorithm>
#include <iterator>
#include <numeric>
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

template <template <typename, typename> typename Container, typename T,
          typename Alloc = std::allocator<T>>
[[nodiscard]] auto argsort(const Container<T, Alloc> &c) {

  auto idx = Container<std::size_t, std::allocator<std::size_t>>(c.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::stable_sort(idx.begin(), idx.end(),
                   [&c](auto lhs, auto rhs) { return c[lhs] < c[rhs]; });

  return idx;
}

template <template <typename, typename> typename Container, typename T,
          typename I, typename TA = std::allocator<T>,
          typename IA = std::allocator<I>>
void reorder(Container<I, IA> order, Container<T, TA> &v) {
  if (order.size() != v.size()) {
    return;
  }

  auto j = I{};
  auto k = I{};
  for (I i = I{}; i < v.size(); i++) {
    while (i != (j = order[i])) {
      k = order[j];
      std::swap(v[j], v[k]);
      std::swap(order[i], order[j]);
    }
  }
}

} // namespace asap

#endif

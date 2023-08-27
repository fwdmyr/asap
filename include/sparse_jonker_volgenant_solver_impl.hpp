#include "sparse_matrix.hpp"
#include <iostream>

namespace asap {

namespace internal {

template <template <typename, typename> typename Container, typename T,
          typename I, typename TA = std::allocator<T>,
          typename IA = std::allocator<I>>
void lapjvsp_update_dual(I &nc, Container<T, TA> &d, Container<T, TA> &v,
                         Container<I, IA> &todo, I &last, T &min_diff) {
  auto j0 = I{0};
  for (I k = last; k < nc; ++k) {
    j0 = todo[k];
    v[j0] += (d[j0] - min_diff);
  }
}

template <template <typename, typename> typename Container, typename I,
          typename IA = std::allocator<I>>
void lapjvsp_update_assignments(Container<I, IA> &lab, Container<I, IA> &y,
                                Container<I, IA> &x, I &j, I &i0) {
  auto i = I{0};
  auto k = I{0};
  while (true) {
    i = lab[j];
    y[j] = i;
    k = j;
    j = x[i];
    x[i] = k;
    if (i == i0) {
      return;
    }
  }
}

template <template <typename, typename> typename Container, typename T,
          typename I, typename TA = std::allocator<T>,
          typename IA = std::allocator<I>>
[[nodiscard]] auto lapjvsp_single_l(
    I l, I nc, Container<T, TA> &d, Container<bool, std::allocator<bool>> &ok,
    Container<I, IA> &free, const Container<I, IA> &first,
    const Container<I, IA> &kk, const Container<T, TA> &cc, Container<T, TA> &v,
    Container<I, IA> &lab, Container<I, IA> &todo, Container<I, IA> &y,
    Container<I, IA> &x, I &td1) {

  static constexpr auto INF = std::numeric_limits<T>::max();

  auto i0 = I{0};
  auto j = I{0};
  auto td2 = I{0};
  auto last = I{0};
  auto j0 = I{0};
  auto i = I{0};
  auto tp = I{0};
  auto min_diff = T{0.0};
  auto dj = T{0.0};
  auto h = T{0.0};
  auto vj = T{0.0};

  for (I jp = 0; jp < nc; ++jp) {
    d[jp] = INF;
    ok[jp] = false;
  }
  min_diff = INF;
  i0 = free[l];

  for (I t = first[i0]; t < first[i0 + 1]; ++t) {
    j = kk[t];
    dj = cc[t] - v[j];
    d[j] = dj;
    lab[j] = i0;
    if (dj <= min_diff) {
      if (dj < min_diff) {
        td1 = -1;
        min_diff = dj;
      }
      ++td1;
      todo[td1] = j;
    }
  }
  for (I hp = 0; hp < td1 + 1; ++hp) {
    j = todo[hp];
    if (y[j] == -1) {
      lapjvsp_update_assignments(lab, y, x, j, i0);
      return td1;
    }
    ok[j] = true;
  }
  td2 = nc - 1;
  last = nc;

  while (true) {
    if (td1 < 0) {
      throw std::runtime_error("No full matching exists");
    }
    j0 = todo[td1];
    --td1;
    i = y[j0];
    todo[td2] = j0;
    --td2;
    tp = first[i];
    while (kk[tp] != j0) {
      ++tp;
    }
    h = cc[tp] - v[j0] - min_diff;

    for (I t = first[i]; t < first[i + 1]; ++t) {
      j = kk[t];
      if (!ok[j]) {
        vj = cc[t] - v[j] - h;
        if (vj < d[j]) {
          d[j] = vj;
          lab[j] = i;
          if (vj == min_diff) {
            if (y[j] == -1) {
              lapjvsp_update_dual(nc, d, v, todo, last, min_diff);
              lapjvsp_update_assignments(lab, y, x, j, i0);
              return td1;
            }
            ++td1;
            todo[td1] = j;
            ok[j] = true;
          }
        }
      }
    }

    if (td1 == -1) {
      min_diff = INF;
      last = td2 + 1;

      for (I jp = 0; jp < nc; ++jp) {
        if ((d[jp] != INF) && (d[jp] <= min_diff) && !ok[jp]) {
          if (d[jp] < min_diff) {
            td1 = -1;
            min_diff = d[jp];
          }
          ++td1;
          todo[td1] = jp;
        }
      }
      for (I hp = 0; hp < td1 + 1; ++hp) {
        j = todo[hp];
        if (y[j] == -1) {
          lapjvsp_update_dual(nc, d, v, todo, last, min_diff);
          lapjvsp_update_assignments(lab, y, x, j, i0);
          return td1;
        }
        ok[j] = true;
      }
    }
  }
}

template <template <typename, typename> typename Container, typename T,
          typename I, typename TA = std::allocator<T>,
          typename IA = std::allocator<I>>
[[nodiscard]] auto lapjvsp(const Container<I, IA> &first,
                           const Container<I, IA> &kk,
                           const Container<T, TA> &cc, I nr, I nc) {

  static constexpr auto INF = std::numeric_limits<T>::max();

  auto l0 = I{0};
  auto jp = I{0};
  auto i = I{0};
  auto lp = I{0};
  auto j1 = I{0};
  auto tp = I{0};
  auto j0p = I{0};
  auto j1p = I{0};
  auto l0p = I{0};
  auto h = I{0};
  auto i0 = I{0};
  auto td1 = I{0};
  auto min_diff = T{0.0};
  auto v0 = T{0.0};
  auto vj = T{0.0};
  auto dj = T{0.0};
  auto v = Container<T, TA>(nc, T{0.0});
  auto x = Container<I, IA>(nr, I{-1});
  auto y = Container<I, IA>(nc, I{-1});
  auto u = Container<T, TA>(nr, T{0.0});
  auto d = Container<T, TA>(nc, T{0.0});
  auto ok = Container<bool, std::allocator<bool>>(nc, false);
  auto xinv = Container<bool, std::allocator<bool>>(nr, false);
  auto free = Container<I, IA>(nr, I{-1});
  auto todo = Container<I, IA>(nc, I{-1});
  auto lab = Container<I, IA>(nc, I{0});

  if (nr == nc) {
    for (I z = 0; z < nc; ++z) {
      v[z] = INF;
    }
    for (I z = 0; z < nr; ++z) {
      for (I t = first[z]; t < first[z + 1]; ++t) {
        jp = kk[t];
        if (cc[t] < v[jp]) {
          v[jp] = cc[t];
          y[jp] = z;
        }
      }
    }
    for (I z = nc - 1; z >= 0; --z) {
      i = y[z];
      if (i == -1) {
        throw std::runtime_error("No full matching exists");
      }
      if (x[i] == -1) {
        x[i] = z;
      } else {
        y[z] = -1;
        xinv[i] = true;
      }
    }
    lp = 0;
    for (I z = 0; z < nr; ++z) {
      if (xinv[z]) {
        continue;
      }
      if (x[z] != -1) {
        min_diff = INF;
        j1 = x[z];
        for (I t = first[z]; t < first[z + 1]; ++t) {
          jp = kk[t];
          if (jp != j1) {
            if (cc[t] - v[jp] < min_diff) {
              min_diff = cc[t] - v[jp];
            }
          }
        }
        u[z] = min_diff;
        tp = first[z];
        while (kk[tp] != j1) {
          ++tp;
        }
        v[j1] = cc[tp] - min_diff;
      } else {
        free[lp] = z;
        ++lp;
      }
    }
    for (I _ = 0; _ < 2; ++_) {
      h = 0;
      l0p = lp;
      lp = 0;
      while (h < l0p) {
        i = free[h];
        ++h;
        j0p = -1;
        j1p = -1;
        v0 = INF;
        vj = INF;
        for (I t = first[i]; t < first[i + 1]; ++t) {
          jp = kk[t];
          dj = cc[t] - v[jp];
          if (dj < vj) {
            if (dj >= v0) {
              vj = dj;
              j1p = jp;
            } else {
              vj = v0;
              v0 = dj;
              j1p = j0p;
              j0p = jp;
            }
          }
        }
        if (j0p < 0) {
          throw std::runtime_error("No full matching exists");
        }
        i0 = y[j0p];
        u[i] = vj;
        if (v0 < vj) {
          v[j0p] += (v0 - vj);
        } else if (i0 != -1) {
          j0p = j1p;
          i0 = y[j0p];
        }
        x[i] = j0p;
        y[j0p] = i;
        if (i0 != -1) {
          if (v0 < vj) {
            --h;
            free[h] = i0;
          } else {
            free[lp] = i0;
            ++lp;
          }
        }
      }
    }
    l0 = lp;
  } else {
    l0 = nr;
    for (I z = 0; z < nr; ++z) {
      free[z] = z;
    }
  }
  td1 = -1;
  for (I l = 0; l < l0; ++l) {
    td1 = lapjvsp_single_l(l, nc, d, ok, free, first, kk, cc, v, lab, todo, y,
                           x, td1);
  }
  return x;
}

} // namespace internal

} // namespace asap

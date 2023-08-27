#include "sparse_matrix.hpp"

namespace asap {

template <typename T>
void lapjvsp_update_dual(int &nc, std::vector<T> &d, std::vector<T> &v,
                         std::vector<int> &todo, int &last, T &min_diff) {
  auto j0 = int{0};
  for (std::size_t k = last; k < nc; ++k) {
    j0 = todo[k];
    v[j0] += (d[j0] - min_diff);
  }
}

inline void lapjvsp_update_assignments(std::vector<int> &lab,
                                       std::vector<int> &y, std::vector<int> &x,
                                       int &j, int &i0) {
  auto i = int{0};
  auto k = int{0};
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

template <typename T>
auto lapjvsp_single_l(std::size_t l, int &nc, std::vector<T> &d,
                      std::vector<bool> &ok, std::vector<int> &free,
                      std::vector<int> &first, std::vector<int> &kk,
                      std::vector<T> &cc, std::vector<T> &v,
                      std::vector<int> &lab, std::vector<int> &todo,
                      std::vector<int> &y, std::vector<int> &x, int &td1) {

  static constexpr auto INF = std::numeric_limits<T>::max();

  // auto jp = int{0};
  auto i0 = int{0};
  auto j = int{0};
  // auto t = int{0};
  auto td2 = int{0};
  // auto hp = int{0};
  auto last = int{0};
  auto j0 = int{0};
  auto i = int{0};
  auto tp = int{0};
  auto min_diff = T{0.0};
  auto dj = T{0.0};
  auto h = T{0.0};
  auto vj = T{0.0};

  for (std::size_t jp = 0; jp < nc; ++jp) {
    d[jp] = INF;
    ok[jp] = false;
  }
  min_diff = INF;
  i0 = free[l];

  for (std::size_t t = first[i0]; t < first[i0 + 1]; ++t) {
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
  for (std::size_t hp = 0; hp < td1 + 1; ++hp) {
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

    for (std::size_t t = first[i]; t < first[i + 1]; ++t) {
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

      for (std::size_t jp = 0; jp < nc; ++jp) {
        if ((d[jp] != INF) && (d[jp] <= min_diff) && !ok[jp]) {
          if (d[jp] < min_diff) {
            td1 = -1;
            min_diff = d[jp];
          }
          ++td1;
          todo[td1] = jp;
        }
      }
      for (std::size_t hp = 0; hp < td1 + 1; ++hp) {
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

template <typename T>
auto lapjvsp(const CompressedSparseRowRepresentation<T> &csr) {

  static constexpr auto INF = std::numeric_limits<T>::max();

  auto first = csr.row_ptr();
  auto kk = csr.col_ind();
  auto cc = csr.val();
  auto nr = static_cast<int>(csr.rows());
  auto nc = static_cast<int>(csr.cols());

  auto l0 = int{0};
  auto jp = int{0};
  // auto t = int{0};
  auto i = int{0};
  auto lp = int{0};
  auto j1 = int{0};
  auto tp = int{0};
  auto j0p = int{0};
  auto j1p = int{0};
  auto l0p = int{0};
  auto h = int{0};
  auto i0 = int{0};
  auto td1 = int{0};
  auto min_diff = T{0.0};
  auto v0 = T{0.0};
  auto vj = T{0.0};
  auto dj = T{0.0};
  auto v = std::vector<T>(nc, T{0.0});
  auto x = std::vector<int>(nr, int{-1});
  auto y = std::vector<int>(nc, int{-1});
  auto u = std::vector<T>(nr, T{0.0});
  auto d = std::vector<T>(nc, T{0.0});
  auto ok = std::vector<bool>(nc, false);
  auto xinv = std::vector<bool>(nr, false);
  auto free = std::vector<int>(nr, int{-1});
  auto todo = std::vector<int>(nc, int{-1});
  auto lab = std::vector<int>(nc, int{0});

  if (nr == nc) {
    for (std::size_t z = 0; z < nc; ++z) {
      v[z] = INF;
    }
    for (std::size_t z = 0; z < nr; ++z) {
      for (std::size_t t = first[z]; t < first[z + 1]; ++t) {
        jp = kk[t];
        if (cc[t] < v[jp]) {
          v[jp] = cc[t];
          y[jp] = z;
        }
      }
    }
    for (std::size_t z = nc - 1; z >= 0; --z) {
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
    for (std::size_t z = 0; z < nr; ++z) {
      if (xinv[z]) {
        continue;
      }
      if (x[z] != -1) {
        min_diff = INF;
        j1 = x[z];
        for (std::size_t t = first[z]; t < first[z + 1]; ++t) {
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
    for (std::size_t _ = 0; _ < 2; ++_) {
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
        for (std::size_t t = first[i]; t < first[i + 1]; ++t) {
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
    for (std::size_t z = 0; z < nr; ++z) {
      free[z] = z;
    }
  }
  td1 = -1;
  for (std::size_t l = 0; l < l0; ++l) {
    td1 = lapjvsp_single_l(l, nc, d, ok, free, first, kk, cc, v, lab, todo, y,
                           x, td1);
  }
  return x;
}

} // namespace asap

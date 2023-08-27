#!/usr/bin/python3

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph


def main():
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    mat = csr_matrix((data, indices, indptr), shape=(3, 3))
    print(mat.toarray())
    res = csgraph.min_weight_full_bipartite_matching(mat)
    print(res)


if __name__ == "__main__":
    main()

#!/usr/bin/python3

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph


def main():
    m = csr_matrix([[1, 2], [3, 0], [0, 4]])
    mT = m.T
    assert np.array_equal(m.data, mT.data)
    assert np.array_equal(m.indptr, mT.indptr)
    assert np.array_equal(m.indices, mT.indices)

    i, j = m.shape
    a = np.arange(np.min(m.shape))
    if not np.all(m.data):
        print('explicit zero weights are removed before matching')

    print()

    m = csr_matrix([[1, 2, 0], [0, 0, 3]])
    mT = m.T
    assert np.array_equal(m.data, mT.data)
    assert np.array_equal(m.indptr, mT.indptr)
    assert np.array_equal(m.indices, mT.indices)

    biadjacency = csr_matrix([[0, 1, 1], [0, 2, 3]])
    row_ind, col_ind = csgraph.min_weight_full_bipartite_matching(biadjacency)
    print(row_ind, col_ind)


if __name__ == "__main__":
    main()

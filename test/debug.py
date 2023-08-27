#!/usr/bin/python3

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph


def main():
    m = csr_matrix([[1, 2], [3, 0], [0, 4]])
    print(m.data)
    print(m.indptr)
    print(m.indices)
    mT = m.T
    assert np.array_equal(m.data, mT.data)
    assert np.array_equal(m.indptr, mT.indptr)
    assert np.array_equal(m.indices, mT.indices)

    print()

    m = csr_matrix([[1, 2, 0], [0, 0, 3]])
    print(m.data)
    print(m.indptr)
    print(m.indices)
    mT = m.T
    assert np.array_equal(m.data, mT.data)
    assert np.array_equal(m.indptr, mT.indptr)
    assert np.array_equal(m.indices, mT.indices)


if __name__ == "__main__":
    main()

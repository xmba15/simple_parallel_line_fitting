#!/usr/bin/env python
import numpy as np


__all__ = ["estimate", "is_inlier"]


def least_square(A, dim: int):
    """
    solve Ax=0 with normalization constraint on x
    """
    height, width = A.shape[:2]

    assert width >= dim + 1, "not enough unknowns"
    assert height >= dim, "not enough equations"

    R = np.linalg.qr(A, mode="r")
    _, _, V = np.linalg.svd(R[width - dim :, width - dim :])
    n = V.T[:, dim - 1]
    c = -np.linalg.inv(R[0 : width - dim, 0 : width - dim]).dot(R[0 : width - dim, width - dim : width].dot(n))

    # n0*x+n1*y+c0=0
    # n0*x+n1*y+c1=0
    # n0*n0+n1*n1=1
    return *n, *c


def augment(xys_list):
    assert len(xys_list) == 2
    for i in range(2):
        assert xys_list[i].ndim == 2 and xys_list[i].shape[1] == 2

    A0 = np.hstack(
        (
            np.ones((xys_list[0].shape[0], 1), dtype=xys_list[0].dtype),
            np.zeros((xys_list[0].shape[0], 1), dtype=xys_list[0].dtype),
            xys_list[0],
        )
    )
    A1 = np.hstack(
        (
            np.zeros((xys_list[1].shape[0], 1), dtype=xys_list[1].dtype),
            np.ones((xys_list[1].shape[0], 1), dtype=xys_list[1].dtype),
            xys_list[1],
        )
    )
    return np.vstack((A0, A1))  # Ax = 0


def estimate(xys_list):
    return least_square(augment(xys_list), 2)


def is_inlier(xy, coeffs, threshold):
    """
    xy: point(x,y)
    coeffs: ((n0,n1), c) where n0*x+n1*y+c=0 is the line equation
    """
    x0, y0 = xy
    n0, n1, c = coeffs
    return np.abs(x0 * n0 + y0 * n1 + c) < threshold

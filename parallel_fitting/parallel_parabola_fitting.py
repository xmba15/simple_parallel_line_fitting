#!/usr/bin/env python
import numpy as np


__all__ = ["estimate", "is_inlier", "augment", "least_square"]


def least_square(A, b, epsilon=1e-9):
    """
    solve Ax=b
    """
    height, width = A.shape[:2]

    U, S, V = np.linalg.svd(A)
    V = V.T
    y = np.zeros(min(height, width))
    z = np.dot(U.T, b)
    k = 0

    while k < min(height, width) and S[k] > epsilon:
        y[k] = z[k] / S[k]
        k += 1

    return V.dot(y)


def augment(xys_list):
    assert len(xys_list) == 2
    for i in range(2):
        assert xys_list[i].ndim == 2 and xys_list[i].shape[1] == 2

    b = np.vstack((xys_list[0][:, 1, np.newaxis], xys_list[1][:, 1, np.newaxis]))
    A0 = np.hstack(
        (
            xys_list[0][:, 0, np.newaxis] ** 2,
            xys_list[0][:, 0, np.newaxis],
            np.ones((xys_list[0].shape[0], 1), dtype=xys_list[0].dtype),
            np.zeros((xys_list[0].shape[0], 1), dtype=xys_list[0].dtype),
        )
    )
    A1 = np.hstack(
        (
            xys_list[1][:, 0, np.newaxis] ** 2,
            xys_list[1][:, 0, np.newaxis],
            np.zeros((xys_list[1].shape[0], 1), dtype=xys_list[1].dtype),
            np.ones((xys_list[1].shape[0], 1), dtype=xys_list[1].dtype),
        )
    )
    A = np.vstack((A0, A1))  # Ax = b

    return A, b


def estimate(xys_list):
    return least_square(*augment(xys_list))


def is_inlier(xy, coeffs, threshold):
    """
    xy: point(x,y)
    coeffs: (a,b,c) where a*x**2+b*x+c=0 is the parabola equation
    """
    x0, y0 = xy
    a, b, c = coeffs
    cubic_eq_coeffs = [4 * a ** 2, 6 * a * b, 2 * (b ** 2 + 2 * (c - y0) * a + 1), 2 * b * (c - y0) - 2 * x0]

    roots = np.roots(cubic_eq_coeffs)
    roots = [np.real(elem) for elem in roots if np.isreal(elem)]

    if len(roots) == 0:
        return False

    def squared_distance_func(x):
        return (
            a ** 2 * x ** 4
            + 2 * a * b * x ** 3
            + (b ** 2 + 2 * (c - y0) * a + 1) * x ** 2
            + (2 * b * (c - y0) - 2 * x0) * x
            + ((c - y0) ** 2 + x0 ** 2)
        )

    distances = squared_distance_func(np.array((roots)))
    distances = np.sort(distances)

    return distances[0] < threshold * threshold

#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from parallel_fitting.ransac import run_ransac
from parallel_fitting.parallel_parabola_fitting import least_square, estimate, is_inlier, augment


def create_parabola_points(coeffs, t_range=np.arange(0, 20, 0.5), noise_range=[0, 0.5], random_seed=2021):
    assert len(coeffs) == 3
    np.random.seed(random_seed)

    xys = np.zeros((len(t_range), 2), dtype=t_range.dtype)
    xys[:, 0] = t_range
    noise = np.random.normal(*noise_range, t_range.shape)
    xys[:, 1] = coeffs[0] * t_range ** 2 + coeffs[1] * t_range + coeffs[2] + noise

    return xys


def create_data():
    direction = np.array([[3.2, 6.7]])
    direction /= np.linalg.norm(direction) + 1e-10

    a = 0.5
    b = 1.0
    c0 = 5
    c1 = 30
    xys_list = [None] * 2
    xys_list[0] = create_parabola_points(
        (a, b, c0), t_range=np.arange(-15, 15, 0.8), noise_range=[-0.4, 3], random_seed=2021
    )

    xys_list[1] = create_parabola_points(
        (a, b, c1), t_range=np.arange(-15, 15, 0.8), noise_range=[-1.0, 1.5], random_seed=2022
    )
    return xys_list


def main():
    xys_list = create_data()
    A, b = augment(xys_list)
    a, b, c0, c1 = estimate(xys_list)

    max_iterations = 100
    sample_size = int(min(len(xys_list[0]), len(xys_list[1])) * 0.3)
    inlier_thresh = 0.05

    start = time.time()
    best_model, _ = run_ransac(
        xys_list,
        estimate,
        lambda xy, coeffs: is_inlier(xy, coeffs, inlier_thresh),
        sample_size,
        sample_size * 2,
        max_iterations,
        stop_at_goal=True,
    )
    a, b, c0, c1 = best_model

    print("\nruntime: {}[ms]\n".format((time.time() - start) * 1e3))

    colors = ["pink", "blue"]
    for (xys, color) in zip(xys_list, colors):
        plt.scatter(xys[:, 0], xys[:, 1], c=color)

    functions = [lambda x: a * x ** 2 + b * x + c0, lambda x: a * x ** 2 + b * x + c1]
    labels = ["first_parabola", "second_parabola"]
    for (xys, function, label) in zip(xys_list, functions, labels):
        plt.plot(xys[:, 0], function(xys[:, 0]), label=label)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import time
import numpy as np
import matplotlib.pyplot as plt
from parallel_fitting.ransac import run_ransac
from parallel_fitting.parallel_line_fitting import estimate, is_inlier


def create_line_points(start, direction, t_range=np.arange(0, 20, 0.5), noise_range=[0, 0.5], random_seed=2021):
    assert start.shape == (1, 2)
    assert direction.shape == (1, 2)
    np.random.seed(random_seed)

    range_arr = np.hstack((t_range[:, np.newaxis], t_range[:, np.newaxis]))
    points = start + direction * range_arr
    noise = np.random.normal(*noise_range, points.shape)
    return points + noise


def create_data():
    direction = np.array([[3.2, 6.7]])
    direction /= np.linalg.norm(direction) + 1e-10

    xys_list = [None] * 2
    xys_list[0] = create_line_points(
        start=np.array([[20.0, 10.0]]),
        direction=direction,
        t_range=np.arange(0, 100, 1.2),
        noise_range=[-0.4, 3],
        random_seed=2021,
    )

    xys_list[1] = create_line_points(
        start=np.array([[4, 8.9]]),
        direction=direction,
        t_range=np.arange(-50, 50, 1.0),
        noise_range=[-1.0, 1.5],
        random_seed=2022,
    )
    return xys_list


def main():
    xys_list = create_data()
    max_iterations = 100
    sample_size = int(min(len(xys_list[0]), len(xys_list[1])) * 0.3)
    inlier_thresh = 0.1

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
    n = best_model[:2]
    c = best_model[2:]
    print("n {}".format(n))
    print("c {}".format(c))

    print("\nruntime: {}[ms]\n".format((time.time() - start) * 1e3))

    colors = ["pink", "blue"]
    for (xys, color) in zip(xys_list, colors):
        plt.scatter(xys[:, 0], xys[:, 1], c=color)

    functions = [
        lambda x: (-c[0] - n[0] * x) / n[1],
        lambda x: (-c[1] - n[0] * x) / n[1],
    ]
    labels = ["first_line", "second_line"]
    for (xys, function, label) in zip(xys_list, functions, labels):
        plt.plot(xys[:, 0], function(xys[:, 0]), label=label)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
import numpy as np


__all__ = ["run_ransac"]


def run_ransac(
    xys_list, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=2021
):
    assert len(xys_list) == 2
    best_ic = 0
    best_model = None
    np.random.seed(random_seed)

    for i in range(max_iterations):
        sample_xys = [None] * 2
        for j in range(2):
            sample_xys[j] = xys_list[j][np.random.choice(range(len(xys_list[j])), size=sample_size)]
        m = estimate(sample_xys)
        coeffs_list = [(m[0], m[1], m[2]), (m[0], m[1], m[3])]
        ics = [0, 0]

        def check_inlier(line_idx):
            check_inlier_cur_line = np.apply_along_axis(
                lambda xy: is_inlier(xy, coeffs_list[line_idx]), 1, xys_list[line_idx]
            )
            ics[line_idx] += np.count_nonzero(check_inlier_cur_line)

        np.vectorize(check_inlier)([0, 1])
        ic = np.sum(ics)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    return best_model, best_ic

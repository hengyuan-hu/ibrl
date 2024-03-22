from collections import defaultdict
import pickle
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import cv2


def generate_grid(cols, rows, figsize=5, fig_h=None, fig_w=None, squeeze=True):
    if fig_h is None:
        fig = plt.figure(figsize=(cols * figsize, rows * figsize))
    else:
        fig = plt.figure(figsize=(cols * fig_w, rows * fig_h))
    ax = fig.subplots(rows, cols, squeeze=squeeze)
    return fig, ax


def split_into_rows_and_cols(num):
    """factorize num into [row x col]"""
    cols = int(np.ceil(np.sqrt(num)))
    rows = int(np.ceil(num / cols))
    assert rows * cols >= num
    return rows, cols


def plot_images(images, scale: Optional[float] = None):
    # images = [cv2.imread(image_path) for image_path in image_paths]
    num_row, num_col = split_into_rows_and_cols(len(images))
    fig, axes = generate_grid(rows=num_row, cols=num_col, squeeze=False)
    for i, image in enumerate(images):
        r = i // num_col
        c = i % num_col
        if scale is not None:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        axes[r][c].imshow(image)
        axes[r][c].set_title(f"img{i}")

    fig.tight_layout()
    fig.show()


def group_by_seed(log_vals, root, group):
    if root is not None and not root.endswith("/"):
        root = f"{root}/"

    grouped_logs = defaultdict(list)
    for key, xs_vals in log_vals.items():
        if root is not None:
            run_name = key[len(root) :].rsplit("/", 1)[0]
        else:
            run_name = key.split("/")[-2]

        if "seed" not in run_name or not group:
            grouped_logs[run_name].append(xs_vals)
            continue

        splits = run_name.split("_")
        non_seeds = [s for s in splits if "seed" not in s]
        run_name = "_".join(non_seeds)
        grouped_logs[run_name].append(xs_vals)

    run_vals = {}
    # grouped_logs: dict{key, list[tuple[]]}
    for key, seed_xs_vals in grouped_logs.items():
        max_len = max(len(xs_vals[0]) for xs_vals in seed_xs_vals)
        max_xs = []
        means = []
        sems = []

        for i in range(max_len):
            ivals = []
            ix = None
            for xs, vals in seed_xs_vals:
                if len(vals) <= i:
                    continue

                ivals.append(vals[i])
                if ix is None:
                    ix = xs[i]
                else:
                    assert ix == xs[i]

            max_xs.append(ix)
            means.append(np.mean(ivals))
            sems.append(np.std(ivals) / np.sqrt(len(ivals)))
        run_vals[key] = (max_xs, means, sems)

    return run_vals


def plot_curves(
    logs, key, root=None, max_x=None, group=True, fig=None, ax: Optional[plt.Axes] = None
):
    if ax is None:
        fig, ax = generate_grid(1, 1, figsize=9)  # type: ignore
    assert ax is not None

    log_vals = {}
    for log_path in logs:
        log = pickle.load(open(log_path, "rb"))
        # print(log_path)
        # print(log)
        # print(log.keys())
        vals = [item[key] for item in log]
        if "step" in log[0]:
            xs = [item["step"] / 1000 for item in log]
        elif "other/step" in log[0]:
            xs = [item["other/step"] / 1000 for item in log]
        else:
            xs = list(range(len(vals)))
        log_vals[log_path] = (xs, vals)

    run_vals = group_by_seed(log_vals, root, group)
    for run, (xs, means, sems) in run_vals.items():
        means = np.array(means)
        sems = np.array(sems)
        if max_x is not None:
            xs = [x for x in xs if x <= max_x]
            means = means[: len(xs)]
            sems = sems[: len(xs)]
        ax.plot(xs, means, label=run)
        ax.fill_between(xs, y1=means + sems, y2=means - sems, alpha=0.25)
        ax.set_xlabel(f"K-steps")

    ax.legend(fontsize=15)
    ax.set_xlim(xmin=0)
    ax.set_title(key, fontsize=15)

    return fig, ax

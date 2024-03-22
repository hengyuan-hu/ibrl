from contextlib import contextmanager
from collections import defaultdict
import time
import numpy as np
import tabulate


class Stopwatch:
    """stop watch in MS"""

    def __init__(self):
        self.times = defaultdict(list)
        self.reset_time = time.time()

        self.init_time = time.time()

    @property
    def total_time(self):
        return time.time() - self.init_time

    @property
    def elapsed_time_since_reset(self):
        return time.time() - self.reset_time

    def reset(self):
        self.times = defaultdict(list)
        self.reset_time = time.time()

    @contextmanager
    def time(self, key):
        t = time.time()
        yield

        self.times[key].append(1000 * (time.time() - t))  # record in ms

    def summary(self, reset=True):
        headers = ["name", "num", "t/call (ms)", "%"]
        total = 0
        times = {}
        for k, v in self.times.items():
            sum_t = np.sum(v)
            mean_t = sum_t / len(v)
            times[k] = (len(v), sum_t, mean_t)
            total += sum_t

        print("Timer Info:")
        rows = []
        for k, (num, sum_t, mean_t) in times.items():
            rows.append([k, f"{num:.1f}", f"{mean_t:.1f}", f"{100 * sum_t / total:.1f}"])

        rows.append(["total(s)", 1, f"{total/1000:.1f}", f"{100 * total / total:.1f}"])
        print(tabulate.tabulate(rows, headers=headers, tablefmt="orgtbl"))

        if reset:
            self.reset()

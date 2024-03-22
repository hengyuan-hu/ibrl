import os
import pickle
from collections import defaultdict
from datetime import datetime
import wandb


class ValueStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counter = 0
        self.summation = 0.0
        self.max_value = -1e38
        self.min_value = 1e38
        self.max_idx = -1
        self.min_idx = -1

    def append(self, v, count=1):
        self.summation += v
        self.counter += count

        if v > self.max_value:
            self.max_value = v
            self.max_idx = self.counter
        if v < self.min_value:
            self.min_value = v
            self.min_idx = self.counter

    def mean(self):
        if self.counter == 0:
            print("Counter %s is 0")
            assert False
        return self.summation / self.counter

    def sum(self):
        return self.summation

    def summary(self, info=None):
        info = "" if info is None else info
        if self.counter > 1:
            return "%s[%5d]: avg: %8.4f, min: %8.4f[%4d], max: %8.4f[%4d]" % (
                info,
                self.counter,
                self.summation / self.counter,
                self.min_value,
                self.min_idx,
                self.max_value,
                self.max_idx,
            )
        elif self.counter == 1:
            prefix = f"{info}: "
            if isinstance(self.min_value, int):
                return prefix + f"{self.min_value}"
            else:
                return prefix + f"{self.min_value:.2f}"
        else:
            return "%s[0]" % (info)


class MultiCounter:
    def __init__(
        self,
        root,
        use_wandb=False,
        *,
        wb_exp_name=None,
        wb_run_name=None,
        wb_group_name=None,
        config=None,
    ):
        self.stats = defaultdict(lambda: ValueStats())
        self.last_time = datetime.now()
        self.max_key_len = 0
        self.pikl_path = os.path.join(root, "log.pkl")
        self.history = []

        self.use_wandb = use_wandb
        if use_wandb:
            os.environ["WANDB_INIT_TIMEOUT"] = "300"
            wandb.init(
                # set the wandb project where this run will be logged
                project=wb_exp_name,
                name=wb_run_name,
                group=wb_group_name,
                config={} if config is None else config,
            )

    def __getitem__(self, key):
        if len(key) > self.max_key_len:
            self.max_key_len = len(key)

        return self.stats[key]

    def append(self, metrics):
        for k, v in metrics.items():
            self[k].append(v)

    def reset(self):
        for k in self.stats.keys():
            self.stats[k].reset()

        self.last_time = datetime.now()

    def summary(self, global_counter, *, reset=True, prefix=""):
        assert self.last_time is not None
        time_elapsed = (datetime.now() - self.last_time).total_seconds()
        print("[%d] Time spent = %.2f s" % (global_counter, time_elapsed))

        self.history.append({k: v.mean() for k, v in self.stats.items() if v.counter > 0})
        with open(self.pikl_path, "wb") as f:
            pickle.dump(self.history, f)

        sorted_keys = sorted([k for k, v in self.stats.items() if v.counter > 0])
        for k in sorted_keys:
            v = self.stats[k]
            if v.counter > 1:
                continue
            info = str(global_counter) + ": " + k.ljust(self.max_key_len + 2)
            print(v.summary(info=info))

        for k in sorted_keys:
            v = self.stats[k]
            if v.counter == 1:
                continue
            info = str(global_counter) + ": " + k.ljust(self.max_key_len + 2)
            print(v.summary(info=info))

        if self.use_wandb:
            to_log = {f"{prefix}{k}": v.mean() for k, v in self.stats.items() if v.counter > 0}
            wandb.log(to_log)

        if reset:
            self.reset()

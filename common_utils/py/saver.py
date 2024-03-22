# model saver that saves top-k model
import os
import torch
import pickle
import numpy as np


class TopkSaver:
    def __init__(self, save_dir, topk):
        self.save_dir = save_dir
        self.topk = topk
        self.perfs = []
        self.model_perf = {}

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save(self, state_dict, perf, *, save_latest=False, force_save_name=None, config=None):
        if force_save_name is not None:
            weight_name = os.path.join(self.save_dir, "%s.pt" % force_save_name)
            torch.save(state_dict, weight_name)
            if config is not None:
                pickle.dump(config, open(f"{weight_name}.cfg", "wb"))

        if save_latest:
            weight_name = os.path.join(self.save_dir, "latest.pt")
            torch.save(state_dict, weight_name)
            if config is not None:
                pickle.dump(config, open(f"{weight_name}.cfg", "wb"))

        if perf is None:
            return False

        if len(self.perfs) < self.topk:
            idx = len(self.perfs)
            self.perfs.append(perf)
        else:
            idx = np.argmin(self.perfs)
            if perf < self.perfs[idx]:
                return False
            self.perfs[idx] = perf

        weight_name = os.path.join(self.save_dir, f"model{idx}.pt")
        torch.save(state_dict, weight_name)
        print(f"Saved model to {weight_name}")
        self.model_perf[weight_name] = perf
        if config is not None:
            pickle.dump(config, open(f"{weight_name}.cfg", "wb"))

        return True

    def get_best_model(self):
        model_perf = sorted([(-perf, model) for model, perf in self.model_perf.items()])
        print(f"retuning the best model {model_perf[0][1]} with score {model_perf[0][0]}")
        return model_perf[0][1]

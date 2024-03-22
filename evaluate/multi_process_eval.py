import time
from collections import defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn")

import common_utils
from common_utils import ibrl_utils as utils
from env.robosuite_wrapper import PixelRobosuite


class EvalProc:
    def __init__(self, seeds, process_id, env_params, terminal_queue: mp.Queue):
        self.seeds = seeds
        self.process_id = process_id
        self.env_params: dict = env_params
        self.terminal_queue = terminal_queue
        self.send_queue = mp.Queue()
        self.recv_queue = mp.Queue()

    def start(self):
        env = PixelRobosuite(**self.env_params)

        results = {}
        for seed in self.seeds:
            np.random.seed(seed)
            obs, _ = env.reset()
            success = False
            while not env.terminal:
                # NOTE: obs["obs"] should be a cpu tensor because it
                # is more complicated to move cuda tensors around.
                self.send_queue.put((self.process_id, obs))
                action = self.recv_queue.get()
                obs, _, _, success, _ = env.step(action)

            results[seed] = float(success)

        self.terminal_queue.put((self.process_id, results))
        return


def run_eval(env_params, agent, num_game, num_proc, seed, verbose=True) -> list[float]:
    assert num_game % num_proc == 0
    env_params["device"] = "cpu"  # avoid sending cuda across processes

    game_per_proc = num_game // num_proc
    terminal_queue = mp.Queue()

    eval_procs = []
    for i in range(num_proc):
        seeds = list(range(seed + i * game_per_proc, seed + (i + 1) * game_per_proc))
        eval_procs.append(EvalProc(seeds, i, env_params, terminal_queue))

    put_queues = {i: proc.recv_queue for i, proc in enumerate(eval_procs)}
    get_queues = {i: proc.send_queue for i, proc in enumerate(eval_procs)}

    processes = {i: mp.Process(target=proc.start) for i, proc in enumerate(eval_procs)}
    for _, p in processes.items():
        p.start()

    t = time.time()
    results = {}
    with torch.no_grad(), utils.eval_mode(agent):
        while len(processes) > 0:
            while not terminal_queue.empty():
                term_idx, proc_results = terminal_queue.get()
                results.update(proc_results)
                processes[term_idx].join()
                processes.pop(term_idx)
                get_queues.pop(term_idx)
                put_queues.pop(term_idx)

            obses = defaultdict(list)
            idxs = []
            for _, get_queue in get_queues.items():
                if get_queue.empty():
                    continue
                data = get_queue.get()
                idxs.append(data[0])
                for k, v in data[1].items():
                    obses[k].append(v)

            if len(obses) == 0:
                continue

            batch_obs = {k: torch.stack(v).cuda() for k, v in obses.items()}
            batch_action = agent.act(batch_obs, eval_mode=True)
            for idx, action in zip(idxs, batch_action):
                put_queues[idx].put(action)

    if verbose:
        print(f"total time {time.time() - t:.2f}")
        for seed in sorted(list(results.keys())):
            print(f"seed {seed}: score: {float(results[seed])}")
        print(common_utils.wrap_ruler(""))

    scores = []
    for seed in sorted(list(results.keys())):
        scores.append(results[seed])
    return scores

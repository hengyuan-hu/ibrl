import torch
import numpy as np
from common_utils import Recorder, Stopwatch
from common_utils import ibrl_utils as utils
from env.robosuite_wrapper import PixelRobosuite


def run_eval(
    env_params,
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
    eval_mode=True,
) -> list[float]:
    scores = []
    lens = []
    stopwatch = Stopwatch()
    recorder = None if record_dir is None else Recorder(record_dir)

    env = PixelRobosuite(**env_params)
    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            np.random.seed(seed + episode_idx)
            with stopwatch.time("reset"):
                obs, image_obs = env.reset()

            terminal = False
            while not terminal:
                if recorder is not None:
                    recorder.add(image_obs)

                with stopwatch.time(f"act"):
                    action = agent.act(obs, eval_mode=eval_mode)

                with stopwatch.time("step"):
                    obs, reward, terminal, _, image_obs = env.step(action)

                rewards.append(reward)
                step += 1

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, len: {env.time_step}"
                )

            scores.append(np.sum(rewards))
            if scores[-1] > 0:
                lens.append(env.time_step)

            if recorder is not None:
                recorder.save(f"episode{episode_idx}")

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")
        print(f"average steps for success games: {np.mean(lens)}")
        stopwatch.summary()

    return scores


if __name__ == "__main__":
    import argparse
    import os
    import time
    import common_utils
    import train_bc
    import train_rl
    from multi_process_eval import run_eval as mp_run_eval
    import rich.traceback

    # make logging more beautiful
    rich.traceback.install()

    os.environ["MUJOCO_GL"] = "egl"
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    np.set_printoptions(precision=4, linewidth=100, suppress=True)
    torch.set_printoptions(linewidth=100, sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--include", type=str, default=None)
    parser.add_argument("--mode", type=str, default="bc", help="bc/rl")
    parser.add_argument("--num_game", type=int, default=10)
    parser.add_argument("--record_dir", type=str, default=None)
    parser.add_argument("--mp", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    common_utils.set_all_seeds(args.seed)

    if args.folder is None:
        weights = [args.weight]
    else:
        weights = common_utils.get_all_files(args.folder, ".pt", args.include)

    print(f"files to eval:")
    eval_items = []
    for weight in weights:
        print(weight, f", repeat {args.repeat}")
        if args.mode == "bc":
            agent, _, env_params = train_bc.load_model(weight, "cuda")
        elif args.mode == "rl":
            agent, _, env_params = train_rl.load_model(weight, "cuda")
        else:
            assert False, f"unsupported mode: {args.mode}"

        for _ in range(args.repeat):
            eval_items.append((weight, agent, env_params))
    print(common_utils.wrap_ruler(""))

    if args.record_dir:
        assert len(eval_items) == 1
        if env_params["env_name"] == "TwoArmTransport":
            env_params["camera_names"] = ["agentview", "robot0_eye_in_hand", "robot1_eye_in_hand"]
        else:
            env_params["camera_names"] = ["agentview", "robot0_eye_in_hand"]

    weight_scores = []
    all_scores = []
    for weight, agent, env_params in eval_items:
        t = time.time()
        if args.mp >= 1:
            assert args.record_dir is None
            scores = mp_run_eval(
                env_params, agent, args.num_game, args.mp, args.seed, verbose=args.verbose
            )
        else:
            scores = run_eval(
                env_params,
                agent,
                args.num_game,
                args.seed,
                args.record_dir,
                verbose=args.verbose,
            )
        all_scores.append(scores)
        print(f"weight: {weight}")
        print(f"score: {np.mean(scores)}, time: {time.time() - t:.1f}")
        weight_scores.append((weight, np.mean(scores)))

    if len(weight_scores) > 1:
        weight_scores = sorted(weight_scores, key=lambda x: -x[1])
        scores = []
        for weight, score in weight_scores:
            print(f"{weight} -> {score}")
            scores.append(score)
        print(f"average score: {100 * np.mean(scores):.2f}")
        print(f"max score: {100 * scores[0]:.2f}")
        max_score_per_seed = np.array(all_scores).max(0)
        print(f"max over seed: {100 * np.mean(max_score_per_seed):.2f}")

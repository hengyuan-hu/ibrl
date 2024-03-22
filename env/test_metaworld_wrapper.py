from metaworld_wrapper import PixelMetaWorld
import torch
import matplotlib.pyplot as plt
import numpy as np


def display_images(frames):
    # Utility to display a frame stack horizontally
    # Assumes frames is a tensor of size (3 * N, Height, Width)

    assert len(frames.shape) == 3
    num_images = frames.shape[0] // 3
    frames = frames.reshape(num_images, 3, frames.shape[1], frames.shape[2])
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(frames[i].permute((1, 2, 0)))
        ax.axis("off")
    plt.show()


def test_random_initializations():
    kwargs = dict(
        env_name="Assembly",
        robots="Sawyer",
        episode_length=100,
        action_repeat=2,
        frame_stack=2,
        obs_stack=1,
        device="cpu",
        use_state=True,
    )

    env_a = PixelMetaWorld(**kwargs)
    env_b = PixelMetaWorld(**kwargs)

    obs_a = env_a.reset()[0]
    obs_b = env_b.reset()[0]
    # Initializing twice should produce different initializations
    assert not torch.all(obs_a["state"] == obs_b["state"])

    obs_a_1 = env_a.reset()[0]
    # Resetting should produce a different initialization
    assert not torch.all(obs_a["state"] == obs_a_1["state"])

    env = PixelMetaWorld(**kwargs)
    for seed in range(3):
        np.random.seed(seed)
        obs = env.reset()[0]
        print(obs["state"][-3:])


def test_action_repeat():
    kwargs = dict(
        env_name="Assembly",
        robots="Sawyer",
        episode_length=100,
        frame_stack=1,
        obs_stack=1,
        device="cpu",
        use_state=False,
    )

    env_a = PixelMetaWorld(**kwargs, action_repeat=1)
    env_b = PixelMetaWorld(**kwargs, action_repeat=20)

    obs_a = env_a.reset()[0]
    obs_b = env_b.reset()[0]

    for _ in range(20):
        obs_a_1 = env_a.step([0, 0, 0, 1])[0]
    change_in_gripper_a = (obs_a["prop"] - obs_a_1["prop"])[3]

    obs_b_1 = env_b.step([0, 0, 0, 1])[0]
    change_in_gripper_b = (obs_b["prop"] - obs_b_1["prop"])[3]

    # Taking one gripper-close action (with action repeat = 20)
    # should be same as taking 20 gripper-close actions
    assert torch.isclose(change_in_gripper_a, change_in_gripper_b)


def test_stack_obs():
    kwargs = dict(
        env_name="Assembly", robots="Sawyer", episode_length=100, device="cpu", use_state=True
    )

    env_a = PixelMetaWorld(**kwargs, obs_stack=1, frame_stack=1, action_repeat=1)
    env_b = PixelMetaWorld(**kwargs, obs_stack=2, frame_stack=2, action_repeat=1)
    env_c = PixelMetaWorld(**kwargs, obs_stack=2, frame_stack=2, action_repeat=20)

    obs_a = env_a.reset()
    obs_b = env_b.reset()
    obs_c = env_c.reset()

    # Check for correct shapes in the first value in the obs tuple
    for key in ["obs", "prop", "state"]:
        if key in ["prop", "state"]:
            assert (
                len(obs_a[0][key].shape)
                == len(obs_b[0][key].shape)
                == len(obs_c[0][key].shape)
                == 1
            )
        elif key in ["obs"]:
            assert (
                len(obs_a[0][key].shape)
                == len(obs_b[0][key].shape)
                == len(obs_c[0][key].shape)
                == 3
            )
        obs_b[0][key].shape[0] == 2 * obs_a[0][key].shape[0]
        obs_b[0][key].shape[0] == obs_c[0][key].shape[0]

    # Regardless of the frame stacking, the second value in the obs tuple
    # (all_image_obs) should only contain one image
    assert (
        obs_a[1]["corner2"].shape[0]
        == obs_b[1]["corner2"].shape[0]
        == obs_c[1]["corner2"].shape[0]
        == 3
    )

    # Check that, when we've just reset, the timesteps represented in the stack
    # are all the same
    for key in ["obs", "prop", "state"]:
        half_length = obs_b[0][key].shape[0] // 2
        assert torch.all(obs_b[0][key][:half_length] == obs_b[0][key][half_length:])
        assert torch.all(obs_c[0][key][:half_length] == obs_c[0][key][half_length:])

    # Check that obs stacking and action repeating interact correctly
    for _ in range(20):
        obs_a_1 = env_a.step([0, 0, 0, 1])
    change_in_gripper_a = obs_a[0]["prop"][3] - obs_a_1[0]["prop"][3]

    for _ in range(20):
        obs_b_1 = env_b.step([0, 0, 0, 1])
    change_in_gripper_b = obs_b[0]["prop"][3] - obs_b_1[0]["prop"][7]

    obs_c_1 = env_c.step([0, 0, 0, 1])
    change_in_gripper_c = obs_c_1[0]["prop"][3] - obs_c_1[0]["prop"][4 + 3]

    assert torch.isclose(change_in_gripper_a, change_in_gripper_b)
    assert torch.isclose(change_in_gripper_a, change_in_gripper_c)


def test_episode_length():
    kwargs = dict(
        env_name="Assembly",
        robots="Sawyer",
        episode_length=5,
        frame_stack=2,
        obs_stack=2,
        device="cpu",
        use_state=False,
    )
    env_a = PixelMetaWorld(**kwargs, action_repeat=2)
    env_b = PixelMetaWorld(**kwargs, action_repeat=4)
    env_c = PixelMetaWorld(**kwargs, action_repeat=5)

    env_a.reset()
    env_b.reset()
    env_c.reset()

    # Check that terminal is true after episode_length *outer* steps
    # regardless of action_repeat
    for i in range(10):
        rl_obs, reward, terminal, success, image_obs = env_a.step([0, 0, 0, 0])
        assert terminal == (i >= 4)
        rl_obs, reward, terminal, success, image_obs = env_b.step([0, 0, 0, 0])
        assert terminal == (i >= 4)
        rl_obs, reward, terminal, success, image_obs = env_c.step([0, 0, 0, 0])
        assert terminal == (i >= 4)


def test_end_on_success():
    kwargs = dict(
        env_name="Assembly",
        robots="Sawyer",
        episode_length=100,
        action_repeat=2,
        frame_stack=2,
        obs_stack=2,
        device="cpu",
        use_state=False,
    )
    env = PixelMetaWorld(**kwargs)
    env.reset()
    for i in range(200):
        action = env.get_heuristic_action(clip_action=True)
        rl_obs, reward, terminal, success, image_obs = env.step(action)
        if reward != 0:
            assert reward == 1
            assert terminal == True
            assert success == True
            break
    assert i < 100


def test_multi_camera():
    kwargs = dict(
        env_name="Assembly",
        robots="Sawyer",
        episode_length=100,
        action_repeat=2,
        frame_stack=2,
        obs_stack=2,
        image_size=256,
        rl_image_size=64,
        device="cpu",
        use_state=False,
        camera_names=["corner2", "corner3", "corner", "topview", "gripperPOV"],
        rl_camera="topview",
    )
    env = PixelMetaWorld(**kwargs)
    rl_obs, image_obs = env.reset()

    print("Displaying initial frames...")

    for key in image_obs:
        print("image_obs at camera", key)
        display_images(image_obs[key])

    print("rl_obs")
    display_images(rl_obs["obs"])

    print("Solving the task...")
    for i in range(100):
        action = env.get_heuristic_action(clip_action=True)
        rl_obs, reward, terminal, success, image_obs = env.step(action)
        if reward != 0:
            break

    print("Displaying final frames...")

    for key in image_obs:
        print("image_obs at camera", key)
        display_images(image_obs[key])

    print("rl_obs")
    display_images(rl_obs["obs"])


if __name__ == "__main__":
    tests = [
        test_random_initializations,
        # test_action_repeat,
        # test_stack_obs,
        # test_episode_length,
        # test_end_on_success,
        # test_multi_camera,
    ]
    for test in tests:
        print("Running", test.__name__)
        test()
        print()

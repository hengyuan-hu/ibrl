from dataclasses import dataclass
import numpy as np
import torch


class MockRobot:
    @dataclass
    class State:
        joint_positions = np.zeros(7)
        joint_velocities = np.zeros(7)

    def __init__(self):
        self.policy_running = False
        self.ee_pos = torch.zeros(3)
        self.ee_quat = torch.rand(4)
        self.ee_quat /= (self.ee_quat**2).sum() ** 0.5

    def set_home_pose(self, home_pose):
        print(f"[mock robot]: set home_pose {home_pose}")

    def update_desired_ee_pose(self, new_pos: torch.Tensor, new_quat: torch.Tensor):
        assert self.policy_running

        assert new_pos.shape == self.ee_pos.shape
        assert new_quat.shape == self.ee_quat.shape

        print(f"[mock]: robot update pos: {new_pos}, quat: {new_quat}")
        self.ee_pos = new_pos
        self.ee_quat = new_quat

    def go_home(self, blocking):
        print(f"[mock robot]: go home")

        self.ee_pos = torch.zeros(3)
        self.ee_quat = torch.rand(4)
        self.ee_quat /= (self.ee_quat**2).sum() ** 0.5

    def get_robot_state(self):
        return MockRobot.State()

    def get_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ee_pos, self.ee_quat

    def is_running_policy(self):
        return self.policy_running

    def terminate_current_policy(self):
        self.policy_running = False

    def start_cartesian_impedance(self):
        assert not self.is_running_policy()
        self.policy_running = True


class MockGripper:
    @dataclass
    class Metadata:
        max_width = 0.08

    @dataclass
    class State:
        width: float

    def __init__(self):
        self.metadata = MockGripper.Metadata()
        self.width = self.metadata.max_width

    def goto(self, width, speed, force, blocking):
        assert width <= self.metadata.max_width, f"{width=}, {self.metadata.max_width=}"
        print(f"[mock]: gripper.goto {width:.4f}")
        self.width = width

    def get_state(self):
        return MockGripper.State(width=self.width)

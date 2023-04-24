from os import path 
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium_robotics.utils import mujoco_utils

class SweepMarblesEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, observation_type="PO", reward_type="dense", **kwargs):
        assert observation_type in {"PO", "FO"}
        assert reward_type in {"dense", "sparse"}

        self.observation_type = observation_type
        self.reward_type = reward_type
        xml_file = path.join(
            path.dirname(path.realpath(__file__)), "../assets/marbles/sweep_marbles.xml"
        )
        if observation_type == "PO":
            observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
            )
        elif observation_type == "FO":
            observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64
            )
        super().__init__(
            model_path=xml_file,
            frame_skip=20,
            observation_space=observation_space,
            **kwargs
        )

    def reset_model(self) -> np.ndarray:
        qpos = np.array([0., 0., 0., 0., 0.42, 1. ,0., 0., 0.])
        qpos[2] = np.random.uniform(-0.005, 0.005)
        qpos[3] = np.random.uniform(-0.025, 0.025)
        self.set_state(qpos, self.init_qvel)
        self.initial_marble_pos = mujoco_utils.get_site_xpos(self.model, self.data, "object0").copy()
        obs, _ = self._get_obs()
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        obs, info = self._get_obs()
        # reward function should check if the marble is in the bin.
        # if the marble xy position is within the bin and the z position is within the bin height
        marble_pos = mujoco_utils.get_site_xpos(self.model, self.data, "object0").copy()
        bin_pos =  mujoco_utils.get_site_xpos(self.model, self.data, "bin_site").copy()

        info["marble_bin_xy_dist"] = xy_dist = np.linalg.norm(marble_pos[:2] - bin_pos[:2])
        info["marble_bin_z_dist"] = z_dist = np.abs(marble_pos[2] - bin_pos[2])
        inside_bin = xy_dist < 0.05 and z_dist < 0.05

        if self.reward_type == "sparse":
            reward = float(inside_bin)
        if self.reward_type == "dense":
            reward = float(inside_bin) - 0.1 * xy_dist  - 0.02 * z_dist

        terminated = inside_bin
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        PO: 7 dimensional. Dims 1-3 is the initial marble pos. Dims 4-7 are sweeper xy, xy vel.
        FO: 13 dimensional. PO obs + marble xyz pos, xyz vel. 
        """
        sweeper_y = mujoco_utils.get_joint_qpos(self.model, self.data, "slide_y").copy()
        sweeper_z = mujoco_utils.get_joint_qpos(self.model, self.data, "rotate_z").copy()
        sweeper_y_v = mujoco_utils.get_joint_qvel(self.model, self.data, "slide_y").copy()
        sweeper_z_v = mujoco_utils.get_joint_qvel(self.model, self.data, "rotate_z").copy()

        sweeper_state =  np.concatenate([sweeper_y, sweeper_z, sweeper_y_v, sweeper_z_v])
        if self.observation_type == "PO":
            obs = np.concatenate([self.initial_marble_pos, sweeper_state])
            return obs, {}
        elif self.observation_type == "FO":
            # get marble center and linear velocities
            marble_pos = mujoco_utils.get_joint_qpos(self.model, self.data, "object0:joint")[:3]
            marble_vel = mujoco_utils.get_joint_qvel(self.model, self.data, "object0:joint")[:3]
            obs = np.concatenate([self.initial_marble_pos, sweeper_state, marble_pos, marble_vel])
            return obs, {}

if __name__ == "__main__":
    import gymnasium
    # env = SweepMarblesEnv(observation_type="FO", render_mode="human")
    env = gymnasium.make("SweepMarblesFODense-v0", render_mode="human")
    while True:
        obs = env.reset()
        done = False
        i = 0
        while not done:
            obs, rew, term, trunc, info = env.step([0.9,0])
            # print("step", i, obs[4])
            i += 1
            done = term or trunc
            if i > 50:
                break
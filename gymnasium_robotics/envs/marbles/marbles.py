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

    def __init__(self, observation_type="PO", pixel_ob="state", reward_type="dense", **kwargs):
        """The sweep marbles environment. The goal is to sweep the marbles into the bin.

        Args:
            observation_type (str, optional): If the observation is fully or partially observed. Defaults to "PO".
            pixel_ob (str, optional): If the observation is pixels or low-dim state. Defaults to "state".
            reward_type (str, optional): If the reward function is dense or sparse. Defaults to "dense".
        """
        assert observation_type in {"PO", "FO"}
        assert pixel_ob in {"state", "pixels"}
        assert reward_type in {"dense", "sparse"}

        self.reward_scale = 10 
        self.observation_type = observation_type
        self.pixel_ob = pixel_ob
        self.reward_type = reward_type
        xml_file = path.join(
            path.dirname(path.realpath(__file__)), "../assets/marbles/sweep_marbles.xml"
        )
        if observation_type == "PO":
            if self.pixel_ob == "state":
                observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
                )
            elif self.pixel_ob == "pixels":
                _observation_space = {
                    "initial_view": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.uint8
                    ),
                    "robot_camera": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.uint8
                    ),
                }
                observation_space = spaces.Dict(_observation_space)
        elif observation_type == "FO":
            if self.pixel_ob == "state":
                observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64
                )
            elif self.pixel_ob == "pixels":
                _observation_space = {
                    "initial_view": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.uint8
                    ),
                    "overhead_camera": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.uint8
                    ),
                    "robot_camera": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.uint8
                    ),
                }
                observation_space = spaces.Dict(_observation_space)

        super().__init__(
            model_path=xml_file,
            frame_skip=20,
            observation_space=observation_space,
            camera_name="overhead_camera",
            **kwargs
        )

    def reset_model(self) -> np.ndarray:
        marble_x_noise = np.random.uniform(-0.005, 0.005)
        marble_y_noise = np.random.uniform(-0.025, 0.025)
        if self.pixel_ob == "pixels":
            # first render the initial view where the sweeper is out of the image
            qpos = np.array([-10., 0., 0., 0.2, 0.42, 1. ,0., 0., 0.])
            qpos[2] += marble_x_noise
            qpos[3] += marble_y_noise
            self.set_state(qpos, self.init_qvel)
            self.initial_view = self.mujoco_renderer.render(self.render_mode, camera_name="robot_camera")

        qpos = np.array([0., 0., 0., 0.2, 0.42, 1. ,0., 0., 0.])
        qpos[2] += marble_x_noise
        qpos[3] += marble_y_noise
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
            reward = 10 * float(inside_bin)
        if self.reward_type == "dense":
            reward = 10 * float(inside_bin) - 0.1 * xy_dist  - 0.02 * z_dist
        
        reward *= self.reward_scale

        terminated = inside_bin
        truncated = False

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        State:
            PO: 7 dimensional. Dims 1-3 is the initial marble pos. Dims 4-7 are sweeper xy, xy vel.
            FO: 13 dimensional. PO obs + marble xyz pos, xyz vel. 
        Pixels:
            PO: TODO: decide PO and FO camera viewpoints.
            FO:
        """
        if self.pixel_ob == "state":
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
        elif self.pixel_ob == "pixels":
            obs = {"initial_view": self.initial_view}
            cameras = ["robot_camera"]
            if self.observation_type == "FO":
                cameras += ["overhead_camera"]
            for c in cameras:
                obs[c] = self.mujoco_renderer.render(self.render_mode, camera_name=c)
            return obs, {}


if __name__ == "__main__":
    import gymnasium
    # env = SweepMarblesEnv(observation_type="FO", render_mode="human")
    env = gymnasium.make("SweepMarblesFOPixelsDense-v0", render_mode="rgb_array")
    import imageio
    for i in range(1):
        video = []
        obs, _ = env.reset()
        img = []
        for k, v in obs.items():
            img.append(v)
        img = np.concatenate(img, axis=1)
        video.append(img)
        done = False
        i = 0
        while not done:
            obs, rew, term, trunc, info = env.step([0.4,0])
            img = []
            for k, v in obs.items():
                img.append(v)
            video.append(np.concatenate(img, axis=1))
            i += 1
            done = term or trunc
            if i > 50:
                break
        imageio.mimsave("test.gif", video, fps=10)
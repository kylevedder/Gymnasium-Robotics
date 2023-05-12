import functools
import itertools
from os import path

import mujoco
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
        assert pixel_ob in {"state", "camera", "pixels", "depth"}
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
            elif self.pixel_ob == "camera":
                observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
                )
            elif self.pixel_ob == "pixels":
                _observation_space = {
                    "sweeper_state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                    ),
                    "initial_view": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.uint8
                    ),
                    "robot_camera": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.uint8
                    ),
                }
                observation_space = spaces.Dict(_observation_space)
            elif self.pixel_ob == "depth":
                _observation_space = {
                    "sweeper_state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                    ),
                    "initial_view": spaces.Box(
                        low=0, high=255, shape=(64,64,1), dtype=np.float64
                    ),
                    "robot_camera": spaces.Box(
                        low=0, high=255, shape=(64,64,1), dtype=np.float64
                    ),
                }
                observation_space = spaces.Dict(_observation_space)
        elif observation_type == "FO":
            if self.pixel_ob == "state":
                observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64
                )
            elif self.pixel_ob == "camera":
                observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64
                )
            elif self.pixel_ob == "pixels":
                _observation_space = {
                    "sweeper_state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                    ),
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
            elif self.pixel_ob == "depth":
                _observation_space = {
                    "sweeper_state": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                    ),
                    "initial_view": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.float64
                    ),
                    "overhead_camera": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.float64
                    ),
                    "robot_camera": spaces.Box(
                        low=0, high=255, shape=(64,64,3), dtype=np.float64
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
        marble_x_noise = np.random.uniform(-0.05, 0.05)
        marble_y_noise = np.random.uniform(-0.005, 0.005)
        if self.pixel_ob in {"depth", "pixels"}:
            # first render the initial view where the sweeper is out of the image
            qpos = np.array([-10., 0., 0., 0.2, 0.42, 1. ,0., 0., 0.])
            qpos[2] += marble_x_noise
            qpos[3] += marble_y_noise
            self.set_state(qpos, self.init_qvel)
            self.initial_view = self.mujoco_renderer.render("rgb_array" if self.pixel_ob == "pixels" else "depth_array", camera_name="robot_camera")

        qpos = np.array([0., 0., 0., 0.2, 0.42, 1. ,0., 0., 0.])
        qpos[2] += marble_x_noise
        qpos[3] += marble_y_noise
        self.set_state(qpos, self.init_qvel)
        self.initial_marble_pos = mujoco_utils.get_site_xpos(self.model, self.data, "object0").copy()
        if self.pixel_ob == "camera":
            self.initial_marble_corners = self._get_marble_corners("robot_camera")
            self.prev_marble_corners = self._get_marble_corners("overhead_camera")

        obs, _ = self._get_obs()
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        obs, info = self._get_obs()
        if self.pixel_ob == "camera" and self.observation_type == "FO":
            self.prev_marble_corners = self._get_marble_corners("overhead_camera")
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
        Camera:
            PO: 8 dimensional. Dims 1-4 are the initial marble corners. Dims 5-8 is the sweeper state.
            FO: 16 dimensional. PO obs + marble xy pos, xy vel.
        Pixels:
            PO: TODO: decide PO and FO camera viewpoints.
            FO:
        """
        sweeper_y = mujoco_utils.get_joint_qpos(self.model, self.data, "slide_y").copy()
        sweeper_z = mujoco_utils.get_joint_qpos(self.model, self.data, "rotate_z").copy()
        sweeper_y_v = mujoco_utils.get_joint_qvel(self.model, self.data, "slide_y").copy()
        sweeper_z_v = mujoco_utils.get_joint_qvel(self.model, self.data, "rotate_z").copy()

        sweeper_state =  np.concatenate([sweeper_y, sweeper_z, sweeper_y_v, sweeper_z_v])
        if self.pixel_ob == "state":
            if self.observation_type == "PO":
                obs = np.concatenate([self.initial_marble_pos, sweeper_state])
                return obs, {}
            elif self.observation_type == "FO":
                # get marble center and linear velocities
                marble_pos = mujoco_utils.get_joint_qpos(self.model, self.data, "object0:joint")[:3]
                marble_vel = mujoco_utils.get_joint_qvel(self.model, self.data, "object0:joint")[:3]
                obs = np.concatenate([self.initial_marble_pos, sweeper_state, marble_pos, marble_vel])
                return obs, {}
        elif self.pixel_ob == "camera":
            if self.observation_type == "PO":
                obs = np.concatenate([sweeper_state, self.initial_marble_corners])
                return obs, {}
            elif self.observation_type == "FO":
                marble_corners = self._get_marble_corners("overhead_camera")
                marble_vel = marble_corners - self.prev_marble_corners
                obs = np.concatenate([sweeper_state, self.initial_marble_corners, marble_corners, marble_vel])
                return obs, {}
        elif self.pixel_ob == "pixels":
            obs = {"sweeper_state": sweeper_state, "initial_view": self.initial_view}
            cameras = ["robot_camera"]
            if self.observation_type == "FO":
                cameras += ["overhead_camera"]
            for c in cameras:
                obs[c] = self.mujoco_renderer.render(self.render_mode, camera_name=c)
            return obs, {}
        elif self.pixel_ob == "depth":
            obs = {"sweeper_state": sweeper_state, "initial_view": self.initial_view[..., None]}
            cameras = ["robot_camera"]
            if self.observation_type == "FO":
                cameras += ["overhead_camera"]
            for c in cameras:
                obs[c] = self.mujoco_renderer.render("depth_array", camera_name=c)[..., None]
            return obs, {}

    def _get_marble_corners(self, camera):
        # get object position
        box_pos = self.data.geom_xpos[self.model.geom('object0').id]
        # Get the world coordinates of the box corners
        box_pos = self.data.geom_xpos[self.model.geom('object0').id]
        box_mat = self.data.geom_xmat[self.model.geom('object0').id].reshape(3, 3)
        box_size = self.model.geom_size[self.model.geom('object0').id]
        offsets = np.array([-1, 1]) * box_size[:, None]
        xyz_local = np.stack(list(itertools.product(*offsets))).T
        xyz_global = box_pos[:, None] + box_mat @ xyz_local
        # Camera matrices multiply homogenous [x, y, z, 1] vectors.
        corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
        corners_homogeneous[:3, :] = xyz_global

        # Get the camera matrix.
        m = self._compute_camera_matrix(camera)

        # Project world coordinates into pixel space. See:
        # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
        xs, ys, s = m @ corners_homogeneous
        # x and y are in the pixel coordinate system.
        x = xs / s
        y = ys / s
        marble_corners = np.array([x[0], y[0], x[-1], y[-1]])
        return marble_corners

    @functools.lru_cache(maxsize=128)
    def _compute_camera_matrix(self, camera_name):
        """Returns the 3x4 camera matrix."""
        camera = self.data.camera(camera_name)
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        pos = camera.xpos.copy()
        rot = camera.xmat.reshape(3,3).T.copy()
        fov = self.model.cam_fovy[camera_id]

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot

        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.width - 1) / 2.0
        image[1, 2] = (self.height - 1) / 2.0
        return image @ focal @ rotation @ translation


if __name__ == "__main__":
    import gymnasium
    import imageio
    env = gymnasium.make("SweepMarblesPOCameraDense-v0", render_mode="rgb_array")
    obs = env.reset()



    # env = gymnasium.make("SweepMarblesFODepthDense-v0", render_mode="rgb_array")
    # obs, _ = env.reset()
    # def depth_to_img(depth):
    #     # logic taken from mujoco colab tutorial
    #     # Shift nearest values to the origin.
    #     depth -= depth.min()
    #     # Scale by 2 mean distances of near rays.
    #     depth /= 2*depth[depth <= 1].mean()
    #     # Scale to [0, 255]
    #     pixels = 255*np.clip(depth, 0, 1)
    #     return pixels.astype(np.uint8).squeeze()

    # for k,v in obs.items():
    #     if k != "sweeper_state":
    #         img = depth_to_img(v)
    #         imageio.imwrite(f"{k}_depth.png", img)


    # env = gymnasium.make("SweepMarblesFOPixelsDense-v0", render_mode="rgb_array")
    # import imageio
    # for i in range(1):
    #     video = []
    #     obs, _ = env.reset()
    #     img = []
    #     for k, v in obs.items():
    #         if k != "sweeper_state":
    #             img.append(v)
    #     img = np.concatenate(img, axis=1)
    #     video.append(img)
    #     done = False
    #     i = 0
    #     while not done:
    #         obs, rew, term, trunc, info = env.step([0.4,0])
    #         img = []
    #         for k, v in obs.items():
    #             if k != "sweeper_state":
    #                 img.append(v)
    #         video.append(np.concatenate(img, axis=1))
    #         i += 1
    #         done = term or trunc
    #         if i > 50:
    #             break
    #     imageio.mimsave("test.gif", video, fps=10)
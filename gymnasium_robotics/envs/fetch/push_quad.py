import os
import itertools
import functools

import numpy as np
import mujoco

from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, goal_distance

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "push_quad.xml")


class MujocoFetchPushQuadEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, camera_names=None, reward_type="sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.camera_names = camera_names
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.03,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        # consists of images and proprioception.
        _obs_space = {}
        if isinstance(camera_names, list) and len(camera_names) > 0:
            for c in camera_names:
                _obs_space[c] = spaces.Box(
                    0, 255, shape=(self.height, self.width, 3), dtype="uint8"
                )
        self.observation_space = spaces.Dict(_obs_space)
        EzPickle.__init__(self, camera_names=camera_names, image_size=32, reward_type=reward_type, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.1 - 0.1, 0.95 + 0.1, 0.42])
        return goal.copy()

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = [1.2, 0.85] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _get_obs(self):
        obs = {}
        if hasattr(self, "mujoco_renderer"):
            self._render_callback()
            for c in self.camera_names:
                img = self.mujoco_renderer.render(self.render_mode, camera_name=c)
                obs[c] = img
        else:
            # BaseRobotEnv has called _get_obs to determine observation space dims but mujoco renderer has not been initialized yet.
            # in this case, return an obs dict with arbitrary values for each ey
            # since observation space will be overwritten later.
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            obs["achieved_goal"] = obs["observation"] = img
        return obs

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        info = {
            "is_success": self._is_success(obj0_pos, self.goal),
        }

        terminated = self.compute_terminated(obj0_pos, self.goal, info)
        truncated = self.compute_truncated(obj0_pos, self.goal, info)

        reward = self.compute_reward(obj0_pos, self.goal, info)

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        # removed super.reset call
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, {}

# push into all four quadrants
class MujocoFetchPushQuadHardEnv(MujocoFetchEnv, EzPickle):
    def __init__(self, camera_names=None, reward_type="sparse", action_space_type="object", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.camera_names = camera_names
        self.prev_goal_dist = None
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.03,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        self.action_space_type = action_space_type
        # consists of images and proprioception.
        _obs_space = {"robot": spaces.Box(low=float("-inf"), high=float("inf"), shape=(6,))}
        if isinstance(camera_names, list) and len(camera_names) > 0:
            for c in camera_names:
                _obs_space[c] = spaces.Box(
                    0, 255, shape=(self.height, self.width, 3), dtype="uint8"
                )
        self.observation_space = spaces.Dict(_obs_space)
        EzPickle.__init__(self, camera_names=camera_names, reward_type=reward_type, action_space_type=action_space_type, **kwargs)
        
        self.goal_idx = 0
        q1_goal = np.array([1.1, 0.95, 0.42])
        q2_goal = np.array([1.1, 0.55, 0.42])
        q3_goal = np.array([1.5, 0.55, 0.42])
        q4_goal = np.array([1.5, 0.95, 0.42])
        self.goals = [q1_goal, q2_goal, q3_goal, q4_goal]

    def _sample_goal(self):
        goal = self.goals[self.goal_idx]
        return goal.copy()

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            # object_xpos = [1.2, 0.85] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_xpos = [1.2, 0.85]
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _get_obs(self):
        obs = {}
        # positions
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

        dt = self.n_substeps * self.model.opt.timestep
        grip_velp = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
        )
        obs["robot"] = np.concatenate([grip_pos, grip_velp], dtype=np.float32)

        if hasattr(self, "mujoco_renderer"):
            self._render_callback()
            for c in self.camera_names:
                img = self.mujoco_renderer.render(self.render_mode, camera_name=c)
                obs[c] = img
        else:
            # BaseRobotEnv has called _get_obs to determine observation space dims but mujoco renderer has not been initialized yet.
            # in this case, return an obs dict with arbitrary values for each ey
            # since observation space will be overwritten later.
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            obs["achieved_goal"] = obs["observation"] = img
        return obs

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        if self.action_space_type == "object":
            # todo: make this xy control, and rescale to 5cm increments.
            action = np.clip(action, -1, 1)
            action = action * 0.05
            obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            obj0_new_pos = obj0_pos + np.array([*action[:2], 0])
            obj0_qpos = np.array(self._utils.get_joint_qpos(self.model, self.data, "object0:joint"))
            obj0_qpos[:2] = obj0_new_pos[:2]
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", obj0_qpos
            )
            self._mujoco.mj_forward(self.model, self.data)
            self._mujoco_step(action)
            self._step_callback()
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self._set_action(action)

            self._mujoco_step(action)

            self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        info = {
            "is_success": self._is_success(obj0_pos, self.goal),
        }

        terminated = self.compute_terminated(obj0_pos, self.goal, info)
        truncated = self.compute_truncated(obj0_pos, self.goal, info)

        if self.reward_type == "dense":
            curr_goal_dist = goal_distance(obj0_pos, self.goal)
            reward = self.compute_reward(curr_goal_dist, self.prev_goal_dist, info)

            if info["is_success"] and self.goal_idx < len(self.goals) - 1:
                self.goal_idx += 1
                self.goal = self.goals[self.goal_idx]

            self.prev_goal_dist = goal_distance(obj0_pos, self.goal)
        else:
            if self.action_space_type == "object":
                curr_goal_dist = goal_distance(obj0_pos, self.goal)
                reward = 1.0 * (curr_goal_dist < 0.05) 
            else:
                curr_goal_dist = goal_distance(obj0_pos, self.goal)
                grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
                # keep gripper somewhat close to the object.
                curr_grip_dist = goal_distance(grip_pos, obj0_pos)
                reward = 1.0 * (curr_goal_dist < 0.05) - 0.01 * curr_grip_dist

            if info["is_success"] and self.goal_idx < len(self.goals) - 1:
                self.goal_idx += 1
                self.goal = self.goals[self.goal_idx]

        return obs, reward, terminated, truncated, info
    
    def compute_reward(self, curr_goal_dist, prev_goal_dist, info):
        # we want prev_goal_dist > curr_goal_dist.
        reward = self.prev_goal_dist - curr_goal_dist
        return reward

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        # removed super.reset call
        self.goal_idx = 0
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        # initialize block distance
        obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        self.prev_goal_dist = goal_distance(obj0_pos, self.goal)
        return obs, {}

    def render(self):
        """Render a frame of the MuJoCo simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        self._render_callback()
        img = self.mujoco_renderer.render(self.render_mode, camera_name="camera_under")
        if np.sum(img) == 0:
            import ipdb; ipdb.set_trace()
        return img

    def close(self):
        """Close contains the code necessary to "clean up" the environment.

        Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
        """
        pass
        # if self.mujoco_renderer is not None:
        #     self.mujoco_renderer.close()


# Here we get 2D coordinates from the camera space
class MujocoFetchPushQuadPoseEnv(MujocoFetchEnv, EzPickle):
    # TODO: include po_reward option
    def __init__(self, camera_names=None, reward_type="sparse", action_space_type="object", include_obj_pose=True, **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.camera_names = camera_names
        self.include_obj_pose = include_obj_pose
        self.action_space_type = action_space_type
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.0,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.03,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        # consists of images and proprioception.
        _obs_space = {"robot": spaces.Box(low=float("-inf"), high=float("inf"), shape=(6,))}

        # add goal pos as input
        _obs_space["goal_pos"] = spaces.Box(low=float("-inf"), high=float("inf"), shape=(3,))

        if isinstance(camera_names, list) and len(camera_names) > 0:
            for c in camera_names:
                _obs_space[c] = spaces.Box(-np.inf, np.inf, shape=(2,))
            if self.include_obj_pose:
                _obs_space["object0"] = spaces.Box(-np.inf, np.inf, shape=(3,))
        self._obs_space = _obs_space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(np.sum([np.prod(x.shape) for x in _obs_space.values()]),))
        EzPickle.__init__(self, camera_names=camera_names, reward_type=reward_type, action_space_type=action_space_type, include_obj_pose=include_obj_pose, **kwargs)
        
        self.goal_idx = 0
        q1_goal = np.array([1.1, 0.95, 0.42])
        q2_goal = np.array([1.1, 0.55, 0.42])
        q3_goal = np.array([1.5, 0.55, 0.42])
        q4_goal = np.array([1.5, 0.95, 0.42])
        self.goals = [q1_goal, q2_goal, q3_goal, q4_goal]
        # self.goals = [q1_goal]

    def _sample_goal(self):
        goal = self.goals[self.goal_idx]
        return goal.copy()

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            # object_xpos = [1.2, 0.85] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_xpos = [1.2, 0.85]
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    @functools.lru_cache(maxsize=128)
    def _compute_camera_matrix(self, camera_name):
        """Returns the 3x4 camera matrix."""
        # If the camera is a 'free' camera, we get its position and orientation
        # from the scene data structure. It is a stereo camera, so we average over
        # the left and right channels. Note: we call `self.update()` in order to
        # ensure that the contents of `scene.camera` are correct.
        # renderer.update_scene(data)

        # pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
        # z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
        # y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
        """
        pos = self._physics.data.cam_xpos[camera_id]
        rot = self._physics.data.cam_xmat[camera_id].reshape(3, 3).T
        fov = self._physics.model.cam_fovy[camera_id]

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
        """
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

    def _get_obs(self):
        obs = {}
        # positions
        if self.action_space_type == "robot":
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

            dt = self.n_substeps * self.model.opt.timestep
            grip_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
            )
            obs["robot"] = np.concatenate([grip_pos, grip_velp], dtype=np.float32)
        else: # object only action space doesn't need robot state.
            obs["robot"] = np.zeros((6,), dtype=np.float32)
        
        obs["goal_pos"] = self.goal.copy()
        if hasattr(self, "mujoco_renderer"):
            for c in self.camera_names:
                # get object position
                box_pos = self.data.geom_xpos[self.model.geom('object0').id]
                # Get the world coordinates of the box corners
                box_pos = self.data.geom_xpos[self.model.geom('object0').id]
                box_mat = self.data.geom_xmat[self.model.geom('object0').id].reshape(3, 3)
                box_size = self.model.geom_size[self.model.geom('object0').id]
                # offsets = np.array([-1, 1]) * box_size[:, None]
                # xyz_local = np.stack(list(itertools.product(*offsets))).T
                # xyz_global = box_pos[:, None] + box_mat @ xyz_local
                xyz_global = box_pos[:, None] 

                # Camera matrices multiply homogenous [x, y, z, 1] vectors.
                corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
                corners_homogeneous[:3, :] = xyz_global

                # Get the camera matrix.
                m = self._compute_camera_matrix(c)

                # Project world coordinates into pixel space. See:
                # https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
                xs, ys, s = m @ corners_homogeneous
                # x and y are in the pixel coordinate system.
                x = xs / s
                y = ys / s
                obs[c] = np.array([x[0],y[0]])
                # pixels = self.mujoco_renderer.render(self.render_mode, camera_name=c)
                # import matplotlib.pyplot as plt
                # fig, ax = plt.subplots(1, 1)
                # ax.imshow(pixels)
                # ax.plot(x, y, '+', c='w')
                # ax.set_axis_off()
                # plt.savefig("observation.png")
                # import ipdb; ipdb.set_trace()
            if self.include_obj_pose:
                obs["object0"] = box_pos
            return np.concatenate(list(obs.values()), axis=0)
        else:
            # BaseRobotEnv has called _get_obs to determine observation space dims but mujoco renderer has not been initialized yet.
            # in this case, return an obs dict with arbitrary values for each ey
            # since observation space will be overwritten later.
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            obs["achieved_goal"] = obs["observation"] = img
            return obs


    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        if self.action_space_type == "object":
            # todo: make this xy control, and rescale to 5cm increments.
            action = np.clip(action, -1, 1)
            action = action * 0.05
            obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
            obj0_new_pos = obj0_pos + np.array([*action[:2], 0])
            obj0_qpos = np.array(self._utils.get_joint_qpos(self.model, self.data, "object0:joint"))
            obj0_qpos[:2] = obj0_new_pos[:2]
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", obj0_qpos
            )
            self._mujoco.mj_forward(self.model, self.data)
            self._mujoco_step(action)
            self._step_callback()
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self._set_action(action)

            self._mujoco_step(action)

            self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")
        # TODO: include ground truth reward here
        info = {
            "is_success": self._is_success(obj0_pos, self.goal),
        }

        terminated = False
        truncated = False

        if self.reward_type == "dense":
            reward = -goal_distance(obj0_pos, self.goal)
            if info["is_success"] and self.goal_idx < len(self.goals) - 1:
                self.goal_idx += 1
                self.goal = self.goals[self.goal_idx]
                obs[6:9] = self.goal.copy()
                reward += 10
        else:
            if self.action_space_type == "object":
                curr_goal_dist = goal_distance(obj0_pos, self.goal)
                reward = 1.0 * (curr_goal_dist < 0.05) 
            else:
                curr_goal_dist = goal_distance(obj0_pos, self.goal)
                grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
                # keep gripper somewhat close to the object.
                curr_grip_dist = goal_distance(grip_pos, obj0_pos)
                reward = 1.0 * (curr_goal_dist < 0.05) - 0.01 * curr_grip_dist

            if info["is_success"] and self.goal_idx < len(self.goals) - 1:
                self.goal_idx += 1
                self.goal = self.goals[self.goal_idx]
                obs[6:9] = self.goal.copy()

        info["goal_idx"] = self.goal_idx
        return obs, reward, terminated, truncated, info
    

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        # removed super.reset call
        self.goal_idx = 0
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {"goal_idx": 0}

    def render(self):
        """Render a frame of the MuJoCo simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        self._render_callback()
        img = self.mujoco_renderer.render(self.render_mode, camera_name="camera_under")
        if np.sum(img) == 0:
            import ipdb; ipdb.set_trace()
        return img

    # def close(self):
    #     """Close contains the code necessary to "clean up" the environment.

    #     Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
    #     """
    #     pass

if __name__ == "__main__":
    env = MujocoFetchPushQuadPoseEnv(["camera_q1", "camera_q2", "camera_q3", "camera_q4"], "dense", "object", render_mode="rgb_array", width=100, height=100)
    returns = 0
    # while True:
    for i in range(1):
        obs, _ = env.reset()
        env.render()
        # go to the first corner
        returns = 0
        for i in range(7):
            obs, rew, trunc, term, info =  env.step(np.array([-0.2, 0.2, 0, 0]))
            # env.render()
            print(f"step {i}", rew, info['is_success'], obs[6:9])
            returns += rew

        # go to the 2nd corner
        for i in range(7):
            obs, rew, trunc, term, info =  env.step(np.array([0, -1., 0, 0]))
            # env.render()
            print(f"step {i}", rew, info['is_success'], obs[6:9])
            returns += rew

        # go to the 3rd corner
        for i in range(7):
            obs, rew, trunc, term, info =  env.step(np.array([1, 0., 0, 0]))
            # env.render()
            print(f"step {i}", rew, info['is_success'], obs[6:9])
            returns += rew

        # go to the 4th corner
        for i in range(7):
            obs, rew, trunc, term, info =  env.step(np.array([0, 1, 0, 0]))
            # env.render()
            print(f"step {i}", rew, info['is_success'], obs[6:9])
            returns += rew
        print("return", returns)
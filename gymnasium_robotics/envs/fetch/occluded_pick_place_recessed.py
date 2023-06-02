import os
import itertools
import functools

import numpy as np
import mujoco

from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, goal_distance

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "occluded_pick_place_recessed.xml")


class FetchOccludedPickPlaceRecessedEnv(MujocoFetchEnv, EzPickle):
    metadata = {"render_modes": ["rgb_array", "depth_array"], 'render_fps': 25}
    def __init__(self, camera_names=[], reward_type="sparse", obj_grip_rew_weight=0.1, obj_goal_rew_weight=1.0, obj_range=0.05, model_xml_path=MODEL_XML_PATH, **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            'object0:joint': [1.33, 0.75, 0.42, 1., 0., 0., 0.],
        }
        self.camera_names = camera_names
        workspace_min=np.array([1.3, 0.35, 0.42])
        workspace_max=np.array([1.47, 1.15, 0.7])
        self.workspace_min = workspace_min
        self.workspace_max = workspace_max
        self.initial_qpos = initial_qpos
        self.obj_grip_rew_weight = obj_grip_rew_weight
        self.obj_goal_rew_weight = obj_goal_rew_weight
        MujocoFetchEnv.__init__(
            self,
            model_path=model_xml_path,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=obj_range,
            target_range=0.05,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        self.cube_body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "object0"
        )
        # consists of images and proprioception.
        _obs_space = {}
        if isinstance(camera_names, list) and len(camera_names) > 0:
            for c in camera_names:
                _obs_space[c] = spaces.Box(
                        0, 255, shape=(self.height, self.width, 3), dtype="uint8"
                    ) if self.render_mode == "rgb_array" else \
                    spaces.Box(
                        0, np.inf, shape=(self.height, self.width, 1), dtype="float32"
                    )
        _obs_space["robot_state"] = spaces.Box(-np.inf, np.inf, shape=(10,), dtype="float32")
        _obs_space["touch"] = spaces.Box(-np.inf, np.inf, shape=(2,), dtype="float32")

        self.observation_space = spaces.Dict(_obs_space)
        EzPickle.__init__(self, camera_names=camera_names, image_size=32, reward_type=reward_type, **kwargs)

    def _sample_goal(self):  # Currently constant goal, no randomization
        return self._utils.get_site_xpos(self.model, self.data, "recesscenter")

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = [1.33, 0.75]
            y_offset = self.np_random.uniform(-self.obj_range, self.obj_range)
            object_xpos[1] += y_offset
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
                obs[c] = img[:,:,None] if self.render_mode == 'depth_array' else img

            touch_left_finger = False
            touch_right_finger = False
            obj = "object0"
            l_finger_geom_id = self.model.geom("robot0:l_gripper_finger_link").id
            r_finger_geom_id = self.model.geom("robot0:r_gripper_finger_link").id
            for j in range(self.data.ncon):
                c = self.data.contact[j]
                body1 = self.model.geom_bodyid[c.geom1]
                body2 = self.model.geom_bodyid[c.geom2]
                body1_name = self.model.body(body1).name
                body2_name = self.model.body(body2).name

                if c.geom1 == l_finger_geom_id and body2_name == obj:
                    touch_left_finger = True
                if c.geom2 == l_finger_geom_id and body1_name == obj:
                    touch_left_finger = True

                if c.geom1 == r_finger_geom_id and body2_name == obj:
                    touch_right_finger = True
                if c.geom2 == r_finger_geom_id and body1_name == obj:
                    touch_right_finger = True

            obs["touch"] = np.array([int(touch_left_finger), int(touch_right_finger)]).astype(np.float32)

            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

            dt = self.n_substeps * self.model.opt.timestep
            grip_velp = (
                self._utils.get_site_xvelp(self.model, self.data, "robot0:grip") * dt
            )

            robot_qpos, robot_qvel = self._utils.robot_get_obs(
                self.model, self.data, self._model_names.joint_names
            )
            gripper_state = robot_qpos[-2:]
            gripper_vel = robot_qvel[-2:] * dt # change to a scalar if the gripper is made symmetric
            
            obs["robot_state"] = np.concatenate([grip_pos, grip_velp, gripper_state, gripper_vel]).astype(np.float32)

        else:
            # BaseRobotEnv has called _get_obs to determine observation space dims but mujoco renderer has not been initialized yet.
            # in this case, return an obs dict with arbitrary values for each ey
            # since observation space will be overwritten later.
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8) if self.render_mode == "rgb_array" \
                else np.zeros((self.height, self.width, 1), dtype=np.float32)
            obs["achieved_goal"] = obs["observation"] = img
        return obs

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = action.copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # check if action is out of bounds
        curr_eef_state = self._utils.get_site_xpos(self.model, self.data, 'robot0:grip')
        next_eef_state = curr_eef_state + (action[:3] * 0.05)

        next_eef_state = np.clip(next_eef_state, self.workspace_min, self.workspace_max)
        clipped_ac = (next_eef_state - curr_eef_state) / 0.05
        action[:3] = clipped_ac

        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0")

        # if object xy is within 0.04m and z is within 0.02m, then terminate
        xy_dist = goal_distance(obj0_pos[:2], self.goal[:2])
        xy_success = xy_dist < 0.04 
        z_dist =  np.abs(obj0_pos[2] - self.goal[2])
        z_success =  np.abs(obj0_pos[2] - self.goal[2]) < 0.02
        terminated = xy_success and z_success
        info = {
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "xy_success": xy_success,
            "z_success": z_success,
        }
        # handled by time limit wrapper.
        truncated = self.compute_truncated(obj0_pos, self.goal, info)

        reward = 0
        if terminated:
            reward = 300
        else:
            # object & goal reward
            dist = goal_distance(obj0_pos, self.goal)
            obj_goal_rew = self.obj_goal_rew_weight * (1 - np.tanh(10.0 * dist))
            reward += obj_goal_rew

            # gripper & object reward
            dist = np.linalg.norm(curr_eef_state - obj0_pos)
            obj_grip_rew = self.obj_grip_rew_weight * (1 - np.tanh(10.0 * dist))
            reward += obj_grip_rew

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
        self.goal = self._sample_goal()
        self._mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, {}

    def close(self):
        pass



if __name__ == "__main__":
    # import gymnasium
    # env = gymnasium.make("FOOccludedPickPlaceRecessed-v0")
    env = FetchOccludedPickPlaceRecessedEnv(["behind_camera"], "dense", render_mode="human", width=64, height=64)
    env.reset()

    # Push block in goal
    obs, _ = env.reset()
    for i in range(20):
        obs, rew, term, trunc, info =  env.step(np.array([-0.0, 0.2, -.2, 0]))
    for i in range(200):
        obs, rew, term, trunc, info = env.step(np.array([0.0, -0.2, 0.0, 0.0]))
        if term:
            print('terminated')
            break
    
import numpy as np
from gymnasium import spaces
from typing import Optional

from gymnasium_robotics.envs.shadow_dexterous_hand import (
    MujocoManipulateEnv,
    MujocoPyManipulateEnv,
)

class HORAPrivilegedMujocoManipulateTouchSensorsEnv(MujocoManipulateEnv):
    def __init__(
        self,
        target_position,
        target_rotation,
        target_position_range,
        reward_type,
        initial_qpos={},
        randomize_initial_position=True,
        randomize_initial_rotation=True,
        distance_threshold=0.01,
        rotation_threshold=0.1,
        n_substeps=20,
        relative_control=False,
        ignore_z_target_rotation=False,
        touch_visualisation="on_touch",
        touch_get_obs="sensordata",
        camera_names=None,
        log_image_keys = None,
        **kwargs,
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation

        """
        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]
        self.camera_names = camera_names if camera_names is not None else []
        self.log_image_keys = log_image_keys if log_image_keys is not None else []
        self.initial_object_state = None
        assert len(self.log_image_keys) == len(self.camera_names)

        self._super_constructor_called = False
        super().__init__(
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=target_position_range,
            reward_type=reward_type,
            initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position,
            randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold,
            n_substeps=n_substeps,
            relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
            **kwargs,
        )
        self._super_constructor_called = True

        for (
            k,
            v,
        ) in (
            self._model_names.sensor_name2id.items()
        ):  # get touch sensor site names and their ids
            if "robot0:TS_" in k:
                self._touch_sensor_id_site_id.append(
                    (
                        v,
                        self._model_names.site_name2id[
                            k.replace("robot0:TS_", "robot0:T_")
                        ],
                    )
                )
                self._touch_sensor_id.append(v)

        if self.touch_visualisation == "off":  # set touch sensors rgba values
            for _, site_id in self._touch_sensor_id_site_id:
                self.model.site_rgba[site_id][3] = 0.0
        elif self.touch_visualisation == "always":
            pass
        
        _obs_space = dict(
                # object position (3), quat (4)
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float32"
                ),
                # object position (3), quat (4)
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float32"
                ),
                # robot qpos (24), qvel (24), initial object qpos (7), qvel (6), desired object qpos (7)
                obs=spaces.Box(
                    -np.inf, np.inf, shape=(68,), dtype="float32"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(68,), dtype="float32"
                ),
                # object qpos (7), qvel(6), touch (92)
                priv_info=spaces.Box(
                    -np.inf, np.inf, shape=(105,), dtype="float32"
                ),
                # object qpos (7), qvel (6)
                object=spaces.Box(
                    -np.inf, np.inf, shape=(13,), dtype="float32"
                ),
                # touch sensors
                touch=spaces.Box(
                    0.0, 1.0, shape=(92,), dtype="float32"
                ),
                # success metric
                log_is_success=spaces.Box(
                   low= -np.inf, high=np.inf, dtype="float32"
                ),
            )

        self.history_len = 30
        self.history_buffer = np.zeros((self.history_len, _obs_space["obs"].shape[0]), dtype=np.float32)
        _obs_space["proprio_hist"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.history_buffer.shape, dtype=np.float32
        )

        if len(self.camera_names) > 0:
            for c, log in zip(self.camera_names, self.log_image_keys):
                key = f"log_{c}" if log else c
                _obs_space[key] = spaces.Box(
                        0, 255, shape=(self.height, self.width, 3), dtype="uint8"
                    ) if self.render_mode == "rgb_array" else \
                    spaces.Box(
                        0, np.inf, shape=(self.height, self.width, 1), dtype="float32"
                    )
        self.observation_space = spaces.Dict(_obs_space)

    def _render_callback(self):
        super()._render_callback()
        if self.touch_visualisation == "on_touch":
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                if self.data.sensordata[touch_sensor_id] != 0.0:
                    self.model.site_rgba[site_id] = self.touch_color
                else:
                    self.model.site_rgba[site_id] = self.notouch_color

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        object_qvel = self._utils.get_joint_qvel(self.model, self.data, "object:joint")

        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        touch_values = []  # get touch sensor readings. if there is one, set value to 1

        if self.touch_get_obs == "sensordata":
            touch_values = self.data.sensordata[self._touch_sensor_id]
        elif self.touch_get_obs == "boolean":
            touch_values = self.data.sensordata[self._touch_sensor_id] > 0.0
        elif self.touch_get_obs == "log":
            touch_values = np.log(self.data.sensordata[self._touch_sensor_id] + 1.0)
        observation = np.concatenate(
            [robot_qpos, robot_qvel, object_qvel, achieved_goal, touch_values]
        )
        if not self._super_constructor_called:
            return {
                "observation": observation.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.ravel().copy(),
            }
        else:
            self._render_callback()
            obs = {}
            for c, log in zip(self.camera_names, self.log_image_keys):
                img = self.mujoco_renderer.render(self.render_mode, camera_name=c)
                key = f"log_{c}" if log else c
                obs[key] = img[:,:,None] if self.render_mode == 'depth_array' else img

            if self.initial_object_state is None:
                self.initial_object_state = observation[48:61]
            obs.update({
                "obs": np.concatenate([observation[:48], self.initial_object_state, self.goal.ravel()]).astype(np.float32),
                "observation": np.concatenate([observation[:48], self.initial_object_state, self.goal.ravel()]).astype(np.float32),
                "priv_info": observation[48:].copy().astype(np.float32), # object, touch
                "object": observation[48:61].copy().astype(np.float32),
                "touch": observation[61:].copy().astype(np.float32),
                "achieved_goal": achieved_goal.copy().astype(np.float32), # needed for reward computation.
                "desired_goal": self.goal.ravel().copy().astype(np.float32),
            })
            return obs
    
    #  BaseRobotEnv methods
    # -----------------------------

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (dictionary): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (integer): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state. This is calculated by :meth:`compute_terminated` of `GoalEnv`.
            truncated (boolean): Whether the truncation condition outside the scope of the MDP is satisfied. Timically, due to a timelimit, but
            it is also calculated in :meth:`compute_truncated` of `GoalEnv`.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). In this case there is a single
            key `is_success` with a boolean value, True if the `achieved_goal` is the same as the `desired_goal`.
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        # update history buffer
        self.history_buffer[1:] = self.history_buffer[:-1]
        self.history_buffer[0] = obs["obs"]
        obs["proprio_hist"] = self.history_buffer.copy()

        is_success = self._is_success(obs["achieved_goal"], self.goal),
        info = {"is_success": 0.0}

        # only log success once per episode for HORA
        obs["log_is_success"] = np.zeros((1,), dtype=np.float32)
        if is_success[0] and not self.episodic_success:
            info["is_success"] = 1.0
            obs["log_is_success"] = np.ones((1,), dtype=np.float32) * info["is_success"]
            self.episodic_success = True

        is_success = bool(info["is_success"])
        reward = 10 if is_success else 0

        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        reward += self.compute_reward(obs["achieved_goal"], self.goal, info)

        terminated = False
        # if object is close to ground, terminate episode
        if obs["object"][8] < 0.05:
            terminated = True
            reward = -1000
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset MuJoCo simulation to initial state.

        Note: Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.

        Args:
            seed (optional integer): The seed that is used to initialize the environment's PRNG (`np_random`). Defaults to None.
            options (optional dictionary): Can be used when `reset` is override for additional information to specify how the environment is reset.

        Returns:
            observation (dictionary) : Observation of the initial state. It should satisfy the `GoalEnv` :attr:`observation_space`.
            info (dictionary): This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()

        self.initial_object_state = None
        obs = self._get_obs()

        # clear history buffer.
        self.history_buffer[:] = 0
        self.history_buffer[0] = obs["obs"]
        obs["proprio_hist"] = self.history_buffer.copy()

        self.episodic_success = False

        obs["log_is_success"] = np.zeros((1,), dtype=np.float32)
        if self.render_mode == "human":
            self.render()

        return obs, {}

class PrivilegedMujocoManipulateTouchSensorsEnv(MujocoManipulateEnv):
    def __init__(
        self,
        target_position,
        target_rotation,
        target_position_range,
        reward_type,
        initial_qpos={},
        randomize_initial_position=True,
        randomize_initial_rotation=True,
        distance_threshold=0.01,
        rotation_threshold=0.1,
        n_substeps=20,
        relative_control=False,
        ignore_z_target_rotation=False,
        touch_visualisation="on_touch",
        touch_get_obs="sensordata",
        camera_names=None,
        log_image_keys = None,
        **kwargs,
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation

        """
        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]
        self.camera_names = camera_names if camera_names is not None else []
        self.log_image_keys = log_image_keys if log_image_keys is not None else []
        self.initial_object_state = None
        assert len(self.log_image_keys) == len(self.camera_names)

        self._super_constructor_called = False
        super().__init__(
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=target_position_range,
            reward_type=reward_type,
            initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position,
            randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold,
            n_substeps=n_substeps,
            relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
            **kwargs,
        )
        self._super_constructor_called = True

        for (
            k,
            v,
        ) in (
            self._model_names.sensor_name2id.items()
        ):  # get touch sensor site names and their ids
            if "robot0:TS_" in k:
                self._touch_sensor_id_site_id.append(
                    (
                        v,
                        self._model_names.site_name2id[
                            k.replace("robot0:TS_", "robot0:T_")
                        ],
                    )
                )
                self._touch_sensor_id.append(v)

        if self.touch_visualisation == "off":  # set touch sensors rgba values
            for _, site_id in self._touch_sensor_id_site_id:
                self.model.site_rgba[site_id][3] = 0.0
        elif self.touch_visualisation == "always":
            pass
        
        _obs_space = dict(
                # object position (3), quat (4)
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float64"
                ),
                # object position (3), quat (4)
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=(7,), dtype="float64"
                ),
                # robot qpos (24), qvel (24), initial object qpos (7), qvel (6)
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(61,), dtype="float64"
                ),
                # object qpos (7), qvel (6)
                object=spaces.Box(
                    -np.inf, np.inf, shape=(13,), dtype="float64"
                ),
                # touch sensors
                touch=spaces.Box(
                    0.0, 1.0, shape=(92,), dtype="float64"
                ),
                # success metric
                log_is_success=spaces.Box(
                   low= -np.inf, high=np.inf, dtype="float64"
                ),
            )
        if len(self.camera_names) > 0:
            for c, log in zip(self.camera_names, self.log_image_keys):
                key = f"log_{c}" if log else c
                _obs_space[key] = spaces.Box(
                        0, 255, shape=(self.height, self.width, 3), dtype="uint8"
                    ) if self.render_mode == "rgb_array" else \
                    spaces.Box(
                        0, np.inf, shape=(self.height, self.width, 1), dtype="float32"
                    )

        self.observation_space = spaces.Dict(_obs_space)
        

    def _render_callback(self):
        super()._render_callback()
        if self.touch_visualisation == "on_touch":
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                if self.data.sensordata[touch_sensor_id] != 0.0:
                    self.model.site_rgba[site_id] = self.touch_color
                else:
                    self.model.site_rgba[site_id] = self.notouch_color

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        object_qvel = self._utils.get_joint_qvel(self.model, self.data, "object:joint")

        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        touch_values = []  # get touch sensor readings. if there is one, set value to 1

        if self.touch_get_obs == "sensordata":
            touch_values = self.data.sensordata[self._touch_sensor_id]
        elif self.touch_get_obs == "boolean":
            touch_values = self.data.sensordata[self._touch_sensor_id] > 0.0
        elif self.touch_get_obs == "log":
            touch_values = np.log(self.data.sensordata[self._touch_sensor_id] + 1.0)
        observation = np.concatenate(
            [robot_qpos, robot_qvel, object_qvel, achieved_goal, touch_values]
        )
        if not self._super_constructor_called:
            return {
                "observation": observation.copy(),
                "achieved_goal": achieved_goal.copy(),
                "desired_goal": self.goal.ravel().copy(),
            }
        else:
            self._render_callback()
            obs = {}
            for c, log in zip(self.camera_names, self.log_image_keys):
                img = self.mujoco_renderer.render(self.render_mode, camera_name=c)
                key = f"log_{c}" if log else c
                obs[key] = img[:,:,None] if self.render_mode == 'depth_array' else img

            if self.initial_object_state is None:
                self.initial_object_state = observation[48:61]

            obs.update({
                "observation": np.concatenate([observation[:48], self.initial_object_state]),
                "object": observation[48:61].copy(),
                "touch": observation[61:].copy(),
                "achieved_goal": achieved_goal.copy(), # needed for reward computation.
                "desired_goal": self.goal.ravel().copy(),
            })
            return obs
    
    #  BaseRobotEnv methods
    # -----------------------------

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (np.ndarray): Control action to be applied to the agent and update the simulation. Should be of shape :attr:`action_space`.

        Returns:
            observation (dictionary): Next observation due to the agent actions .It should satisfy the `GoalEnv` :attr:`observation_space`.
            reward (integer): The reward as a result of taking the action. This is calculated by :meth:`compute_reward` of `GoalEnv`.
            terminated (boolean): Whether the agent reaches the terminal state. This is calculated by :meth:`compute_terminated` of `GoalEnv`.
            truncated (boolean): Whether the truncation condition outside the scope of the MDP is satisfied. Timically, due to a timelimit, but
            it is also calculated in :meth:`compute_truncated` of `GoalEnv`.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). In this case there is a single
            key `is_success` with a boolean value, True if the `achieved_goal` is the same as the `desired_goal`.
        """
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
        }
        obs["log_is_success"] = np.ones((1,), dtype=np.float64) * info["is_success"]

        is_success = bool(info["is_success"])
        reward = 10 if is_success else 0

        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)

        reward += self.compute_reward(obs["achieved_goal"], self.goal, info)

        terminated = False
        # if object is close to ground, terminate episode
        if obs["object"][8] < 0.05:
            terminated = True
            reward = -1000
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset MuJoCo simulation to initial state.

        Note: Attempt to reset the simulator. Since we randomize initial conditions, it
        is possible to get into a state with numerical issues (e.g. due to penetration or
        Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        In this case, we just keep randomizing until we eventually achieve a valid initial
        configuration.

        Args:
            seed (optional integer): The seed that is used to initialize the environment's PRNG (`np_random`). Defaults to None.
            options (optional dictionary): Can be used when `reset` is override for additional information to specify how the environment is reset.

        Returns:
            observation (dictionary) : Observation of the initial state. It should satisfy the `GoalEnv` :attr:`observation_space`.
            info (dictionary): This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        super().reset(seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()

        self.initial_object_state = None
        obs = self._get_obs()
        obs["log_is_success"] = np.zeros((1,), dtype=np.float64)
        if self.render_mode == "human":
            self.render()

        return obs, {}

class MujocoManipulateTouchSensorsEnv(MujocoManipulateEnv):
    def __init__(
        self,
        target_position,
        target_rotation,
        target_position_range,
        reward_type,
        initial_qpos={},
        randomize_initial_position=True,
        randomize_initial_rotation=True,
        distance_threshold=0.01,
        rotation_threshold=0.1,
        n_substeps=20,
        relative_control=False,
        ignore_z_target_rotation=False,
        touch_visualisation="on_touch",
        touch_get_obs="sensordata",
        **kwargs,
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation

        """
        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]

        super().__init__(
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=target_position_range,
            reward_type=reward_type,
            initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position,
            randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold,
            n_substeps=n_substeps,
            relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
            **kwargs,
        )

        for (
            k,
            v,
        ) in (
            self._model_names.sensor_name2id.items()
        ):  # get touch sensor site names and their ids
            if "robot0:TS_" in k:
                self._touch_sensor_id_site_id.append(
                    (
                        v,
                        self._model_names.site_name2id[
                            k.replace("robot0:TS_", "robot0:T_")
                        ],
                    )
                )
                self._touch_sensor_id.append(v)

        if self.touch_visualisation == "off":  # set touch sensors rgba values
            for _, site_id in self._touch_sensor_id_site_id:
                self.model.site_rgba[site_id][3] = 0.0
        elif self.touch_visualisation == "always":
            pass

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

    def _render_callback(self):
        super()._render_callback()
        if self.touch_visualisation == "on_touch":
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                if self.data.sensordata[touch_sensor_id] != 0.0:
                    self.model.site_rgba[site_id] = self.touch_color
                else:
                    self.model.site_rgba[site_id] = self.notouch_color

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(
            self.model, self.data, self._model_names.joint_names
        )
        object_qvel = self._utils.get_joint_qvel(self.model, self.data, "object:joint")

        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        touch_values = []  # get touch sensor readings. if there is one, set value to 1

        if self.touch_get_obs == "sensordata":
            touch_values = self.data.sensordata[self._touch_sensor_id]
        elif self.touch_get_obs == "boolean":
            touch_values = self.data.sensordata[self._touch_sensor_id] > 0.0
        elif self.touch_get_obs == "log":
            touch_values = np.log(self.data.sensordata[self._touch_sensor_id] + 1.0)
        observation = np.concatenate(
            [robot_qpos, robot_qvel, object_qvel, achieved_goal, touch_values]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }


class MujocoPyManipulateTouchSensorsEnv(MujocoPyManipulateEnv):
    def __init__(
        self,
        target_position,
        target_rotation,
        target_position_range,
        reward_type,
        initial_qpos={},
        randomize_initial_position=True,
        randomize_initial_rotation=True,
        distance_threshold=0.01,
        rotation_threshold=0.1,
        n_substeps=20,
        relative_control=False,
        ignore_z_target_rotation=False,
        touch_visualisation="on_touch",
        touch_get_obs="sensordata",
        **kwargs,
    ):
        """Initializes a new Hand manipulation environment with touch sensors.

        Args:
            touch_visualisation (string): how touch sensor sites are visualised
                - "on_touch": shows touch sensor sites only when touch values > 0
                - "always": always shows touch sensor sites
                - "off" or else: does not show touch sensor sites
            touch_get_obs (string): touch sensor readings
                - "boolean": returns 1 if touch sensor reading != 0.0 else 0
                - "sensordata": returns original touch sensor readings from self.sim.data.sensordata[id]
                - "log": returns log(x+1) touch sensor readings from self.sim.data.sensordata[id]
                - "off" or else: does not add touch sensor readings to the observation

        """
        self.touch_visualisation = touch_visualisation
        self.touch_get_obs = touch_get_obs
        self._touch_sensor_id_site_id = []
        self._touch_sensor_id = []
        self.touch_color = [1, 0, 0, 0.5]
        self.notouch_color = [0, 0.5, 0, 0.2]

        super().__init__(
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=target_position_range,
            reward_type=reward_type,
            initial_qpos=initial_qpos,
            randomize_initial_position=randomize_initial_position,
            randomize_initial_rotation=randomize_initial_rotation,
            distance_threshold=distance_threshold,
            rotation_threshold=rotation_threshold,
            n_substeps=n_substeps,
            relative_control=relative_control,
            ignore_z_target_rotation=ignore_z_target_rotation,
            **kwargs,
        )

        for (
            k,
            v,
        ) in (
            self.sim.model._sensor_name2id.items()
        ):  # get touch sensor site names and their ids
            if "robot0:TS_" in k:
                self._touch_sensor_id_site_id.append(
                    (
                        v,
                        self.sim.model._site_name2id[
                            k.replace("robot0:TS_", "robot0:T_")
                        ],
                    )
                )
                self._touch_sensor_id.append(v)

        if self.touch_visualisation == "off":  # set touch sensors rgba values
            for _, site_id in self._touch_sensor_id_site_id:
                self.sim.model.site_rgba[site_id][3] = 0.0
        elif self.touch_visualisation == "always":
            pass

        obs = self._get_obs()
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                achieved_goal=spaces.Box(
                    -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float64"
                ),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"
                ),
            )
        )

    def _render_callback(self):
        super()._render_callback()
        if self.touch_visualisation == "on_touch":
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                if self.sim.data.sensordata[touch_sensor_id] != 0.0:
                    self.sim.model.site_rgba[site_id] = self.touch_color
                else:
                    self.sim.model.site_rgba[site_id] = self.notouch_color

    def _get_obs(self):
        robot_qpos, robot_qvel = self._utils.robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel("object:joint")

        achieved_goal = (
            self._get_achieved_goal().ravel()
        )  # this contains the object position + rotation
        touch_values = []  # get touch sensor readings. if there is one, set value to 1

        if self.touch_get_obs == "sensordata":
            touch_values = self.sim.data.sensordata[self._touch_sensor_id]
        elif self.touch_get_obs == "boolean":
            touch_values = self.sim.data.sensordata[self._touch_sensor_id] > 0.0
        elif self.touch_get_obs == "log":
            touch_values = np.log(self.sim.data.sensordata[self._touch_sensor_id] + 1.0)

        observation = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                object_qvel,
                achieved_goal,
                touch_values,
            ]
        )

        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.ravel().copy(),
        }

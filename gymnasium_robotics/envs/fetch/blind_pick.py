import os
import numpy as np
import mujoco
from pathlib import Path
from gymnasium import spaces 
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, goal_distance
from gymnasium_robotics.utils.flow import RAFTWrapper

# for VIP reward.
import cv2
from PIL import Image  
import torch
import torchvision.transforms as T
from vip import load_vip
import torch
from copy import deepcopy
from typing import Dict, List
import tqdm

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "blind_pick.xml")


class FetchBlindPickEnv(MujocoFetchEnv, EzPickle):
    metadata = {"render_modes": ["rgb_array", "depth_array"], 'render_fps': 50}
    render_mode = "rgb_array"
    def __init__(self, camera_names=None, reward_type="dense", obj_range=0.07, include_obj_state=False, n_substeps=20, **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            'object0:joint': [1.33, 0.75, 0.42, 1., 0., 0., 0.],
        }
        self.camera_names = camera_names if camera_names is not None else []
        workspace_min=np.array([1.1, 0.44, 0.42])
        workspace_max=np.array([1.5, 1.05, 0.7])

        self.workspace_min = workspace_min
        self.workspace_max = workspace_max
        self.initial_qpos = initial_qpos
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=n_substeps,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=obj_range,
            target_range=0.0,
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
        self.include_obj_state = include_obj_state
        if include_obj_state:
            _obs_space["obj_state"] = spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32")

        self.observation_space = spaces.Dict(_obs_space)
        EzPickle.__init__(self, camera_names=camera_names, image_size=32, reward_type=reward_type, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.33, 0.75, 0.60])
        return goal.copy()

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position of object.
        if self.has_object:
            object_xpos = [1.3, 0.75]
            # sample in a rectangular region and offset by a random amount
            object_xpos[0] += self.np_random.uniform(-self.obj_range, self.obj_range)
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
            if self.include_obj_state:
                obj0_pos = self._utils.get_site_xpos(self.model, self.data, "object0").copy()
                obs["obj_state"] = obj0_pos.astype(np.float32)

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
        info = {
            "is_success": self._is_success(obj0_pos, self.goal),
        }

        terminated = goal_distance(obj0_pos, self.goal) < 0.05
        # handled by time limit wrapper.
        truncated = self.compute_truncated(obj0_pos, self.goal, info)

        # reward = self.compute_reward(obj0_pos, self.goal, info)
        # success bonus
        reward = 0
        if terminated:
            # print("success phase")
            reward = 300
        else:
            dist = np.linalg.norm(curr_eef_state - obj0_pos)
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward
            # msg = "reaching phase"

            # grasping reward
            if obs["touch"].all():
                reward += 0.25
                dist = np.linalg.norm(self.goal - obj0_pos)
                picking_reward = 1 - np.tanh(10.0 * dist)
                reward += picking_reward
            #     msg = "picking phase"
            # print(msg)

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

    def close(self):
        pass


class VIPRewardBlindPick(FetchBlindPickEnv):

    def __init__(self, image_keys, goal_img_paths, aggregation='mean', device='cpu', **kwargs):

        super().__init__(**kwargs)
        # update observation space dict to have a `log_success` key.
        self.observation_space.spaces['log_success'] = spaces.Box(-np.inf, np.inf, dtype="float32")

        self.image_keys = image_keys
        assert len(image_keys) > 0
        assert len(goal_img_paths) == len(image_keys) # currently assumes 1:1 mapping between image keys to goal images, but can be 1:N.

        self.vip_model = load_vip()
        self.vip_transform = T.Compose([T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor()])
        self.aggregation = aggregation
        self.device = device
        # store the goal embedding.
        self.goal_embeddings = {}
        for image_key, goal_path in zip(image_keys, goal_img_paths):
            directory_path = os.path.dirname(__file__)
            path = os.path.join(directory_path, goal_path)
            img = cv2.imread(path) # BGR format.
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cur = self.vip_transform(Image.fromarray(img)).unsqueeze(0)
            with torch.no_grad():
                embedding = self.vip_model(img_cur.to(self.device))
                self.goal_embeddings[image_key] = embedding.cpu().numpy()

    def reset(self, **kwargs): 
        obs, info = super().reset(**kwargs)
        obs["log_success"] = 0.0
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        reward_per_image = {}
        for k in obs.keys():
            if k not in self.image_keys:
                continue
            img = cv2.cvtColor(deepcopy(obs[k]), cv2.COLOR_RGB2BGR)
            img_cur = self.vip_transform(Image.fromarray(img.astype(np.uint8))).unsqueeze(0)
            with torch.no_grad():
                embeddings = self.vip_model(img_cur.to(self.device))
                embeddings = embeddings.cpu().numpy()
            distance = np.linalg.norm(embeddings - self.goal_embeddings[k])
            reward_per_image[k] = -distance
            # update info with reward per image
            info[f'{k}_reward'] = reward_per_image[k]


        if self.aggregation == 'mean':
            final_reward = np.mean(list(reward_per_image.values()))
        elif self.aggregation == 'max':
            final_reward = np.max(list(reward_per_image.values()))
        elif self.aggregation == 'min':
            final_reward = np.min(list(reward_per_image.values()))

        # give bonus if gripper is closed around block and in the correct position.
        gripper_offset = np.array([0.03, 0, 0.02])
        term = False
        success = False
        if obs["touch"].all() and np.linalg.norm(self.goal - (obs["robot_state"][:3] + gripper_offset)) < 0.05:
            final_reward += 100000
            term = True
            success = True
        obs["log_success"] = float(success)
        return obs, final_reward, term, trunc, info





if __name__ == "__main__":
    # import omegaconf
    # import hydra
    # import torch
    # import torchvision.transforms as T
    # import numpy as np
    # from PIL import Image

    # from vip import load_vip

    cam_keys = ["camera_side", "camera_front", "gripper_camera_rgb"]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    

    raft_wrapper = RAFTWrapper(Path("/RAFT/models/raft-things.pth"), camera_keys=cam_keys, device=device)
    # vip = load_vip()
    # vip.eval()
    # vip.to(device)

    # ## DEFINE PREPROCESSING
    # transforms = T.Compose([T.Resize(256),
    #     T.CenterCrop(224),
    #     T.ToTensor()]) # ToTensor() divides by 255

    # ## ENCODE IMAGE
    # image = np.random.randint(0, 255, (500, 500, 3))
    # preprocessed_image = transforms(Image.fromarray(image.astype(np.uint8))).reshape(-1, 3, 224, 224)
    # preprocessed_image.to(device) 
    # with torch.no_grad():
    #     embedding = vip(preprocessed_image * 255.0) ## vip expects image input to be [0-255]
    # print(embedding.shape) # [1, 1024]

    import imageio
    # env = FetchBlindPickEnv(cam_keys, "dense", render_mode="rgb_array", width=32, height=32, obj_range=0.001)
    env = VIPRewardBlindPick(image_keys=["camera_front"], goal_img_paths=["./blindpick_final_camera_front.png"], device=device,  camera_names=cam_keys, reward_type="dense", render_mode="rgb_array", width=256, height=256, obj_range=0.001, n_substeps=10)

    imgs = []
    obs, _ = env.reset()


    for _ in tqdm.tqdm(range(3)):
        obs,_ = env.reset()
        raft_wrapper(obs)
        for i in range(10):
            
            

            obs, rew, term, trunc, info= env.step(np.array([-0.1,0,-1,-1.0]))
            # obs, *_ = env.step(env.action_space.sample())
            obs = raft_wrapper(obs)
            full_obs_keys = ['camera_side', 'camera_front', 'gripper_camera_rgb', 'camera_side_flow', 'camera_front_flow', 'gripper_camera_rgb_flow']
            concatenated_obs = np.concatenate([obs[k] for k in full_obs_keys], axis=1)
            imgs.append(concatenated_obs)
    

    # Convert from float to uint8
    imgs = np.array(imgs)
    imgs = (imgs * 255).astype(np.uint8)
    imageio.mimwrite("test.mp4", imgs)

    exit(0)

    from collections import defaultdict
    demo = defaultdict(list)
    while True:
        obs, _ = env.reset()
        for k in obs.keys():
            if k in cam_keys:
                demo[k].append(obs[k])
        # open the gripper and descend
        for i in range(10):
            obs, rew, term, trunc, info = env.step(np.array([-0.1, 0.0, -1, 1.0]))
            for k in obs.keys():
                if k in cam_keys:
                    demo[k].append(obs[k])
            # print(rew)
        # close gripper
        for i in range(10):
            obs, rew, term, trunc, info= env.step(np.array([0,0,0.0,-1.0]))
            for k in obs.keys():
                if k in cam_keys:
                    demo[k].append(obs[k])
            print(rew)
        # lift up cube
        for i in range(10):
            obs, rew, term, trunc, info = env.step(np.array([0,0,1.0,-1.0]))
            for k in obs.keys():
                if k in cam_keys:
                    demo[k].append(obs[k])
            print(rew)
            if term:
                # for k in cam_keys:
                #     imageio.imwrite(f'blindpick_final_{k}.png', obs[k])
                break

    # save each key as a mp4 with imageio                
    for k, v in demo.items():
        imageio.mimwrite(f"{k}.mp4", v)

    # import ipdb; ipdb.set_trace()
import os

import numpy as np
import mujoco

from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, goal_distance
from gymnasium_robotics.envs.fetch.occluded_pick_place_recessed import FetchOccludedPickPlaceRecessedEnv


MODEL_XML_PATH = os.path.join("fetch", "occluded_pick_place_recessed_2goal.xml")

class FetchOccludedPickPlaceRecessed2GoalEnv(FetchOccludedPickPlaceRecessedEnv):
    def __init__(self, **kwargs):
        super().__init__(model_xml_path=MODEL_XML_PATH, **kwargs)
        self.goal_indicator_offset = np.array([
            [0.0,-0.075,0.075],  # goal0 offset
            [0.0,0.075,0.075]    # goal1 offset
        ])  

    def _sample_goal(self):  # Randomly select between goal 0 and 1 and place indicator
        goal_num = np.random.randint(2)
        goal_pos = self._utils.get_site_xpos(self.model, self.data, f"recesscenter{goal_num}")
        
        # Move indicator
        body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, "goal_indicator")
        self.model.body_pos[body_id] = goal_pos + self.goal_indicator_offset[goal_num]
        self._mujoco.mj_forward(self.model, self.data)
        return goal_pos

if __name__ == "__main__":
    # import gymnasium
    # env = gymnasium.make("FOOccludedPickPlaceRecessed-v0")
    env = FetchOccludedPickPlaceRecessed2GoalEnv(camera_names=["external_camera_0", "behind_camera"], reward_type="dense", render_mode="human", width=64, height=64)
    obs, _ = env.reset()
    
    # Push block in goal0
    obs, _ = env.reset()
    for i in range(20):
        obs, rew, term, trunc, info =  env.step(np.array([-0.0, 0.2, -.2, 0]))
    for i in range(200):
        obs, rew, term, trunc, info = env.step(np.array([0.0, -0.2, 0.0, 0.0]))
        if term:
            print('terminated')
            break


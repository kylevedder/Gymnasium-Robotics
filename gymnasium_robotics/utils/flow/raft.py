from pathlib import Path
from typing import Dict, List
import numpy as np
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz

import torch

class RAFTArgs:

    def __init__(self) -> None:
        self.small = False
        self.mixed_precision = False

    def __contains__(self, key):
        return False
    
class RAFTWrapper:

    def __init__(self, model_checkpoint : Path, camera_keys : List[str], device = 'cuda') -> None:
        self.model = torch.nn.DataParallel(RAFT(RAFTArgs()))
        self.model.load_state_dict(torch.load(model_checkpoint))
        self.model = self.model.module
        self.model.to(device)
        self.model.eval()
        self.prior_obs_dict = None
        self.camera_keys = camera_keys
        self.device = device

    def __call__(self, obs_dict : Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        def key_to_flow_key(k : str):
            return k + "_flow"

        if self.prior_obs_dict is None:
            self.prior_obs_dict = obs_dict
            return dict({key_to_flow_key(k) : np.zeros_like(obs_dict[k]) for k in self.camera_keys}, **obs_dict)
        
        def image_to_tensor(img : np.ndarray):
            torch_img = torch.from_numpy(img.copy())
            torch_img = torch_img.permute(2, 0, 1)
            torch_img = torch_img.float()
            torch_img = torch_img[None]
            return torch_img.to(self.device)
        
        def process_key(k):
            prior_obs = self.prior_obs_dict[k]
            curr_obs = obs_dict[k]

            prior_obs_torch = image_to_tensor(prior_obs)
            curr_obs_torch = image_to_tensor(curr_obs)
            # forward pass through RAFT
            # with nograd
            with torch.no_grad():

                flow_low, flow_up = self.model(prior_obs_torch, curr_obs_torch, iters=20, test_mode=True)

            flow_up_img = flow_up[0].permute(1, 2, 0).cpu().numpy()
            flow_up_img = flow_viz.flow_to_image(flow_up_img)
            return flow_up_img
        

        flow_res_dict = { key_to_flow_key(k) : process_key(k) for k in self.camera_keys }
        self.prior_obs_dict = obs_dict
        return dict(flow_res_dict, **obs_dict)
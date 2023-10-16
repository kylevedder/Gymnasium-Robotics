import cv2
from typing import List

class Resizer:
    def __init__(self, resize_scale : int, keys_to_modify : List[str]):
        assert isinstance(resize_scale, int), f"Resize scale must be an integer, got {type(resize_scale)}"
        if not (resize_scale > 0 and (resize_scale & (resize_scale - 1) == 0)):
            raise ValueError("Resize scale must be a power of 2")
        self.resize_scale = resize_scale
        self.keys_to_modify = keys_to_modify

    def __call__(self, images_dict):
        resized_images_dict = {}
        for key, image in images_dict.items():
            if key in self.keys_to_modify:
                resized_image = cv2.resize(image, (image.shape[1] // self.resize_scale, image.shape[0] // self.resize_scale))
            else:
                resized_image = image
            resized_images_dict[key] = resized_image
        return resized_images_dict

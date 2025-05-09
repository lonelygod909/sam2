import logging
import random
from copy import deepcopy

import numpy as np

import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.matting_raw_dataset import VideoMatteRawDataset
from training.dataset.matting_sampler import MattingSampler

from training.utils.data_utils_matting import Frame, MattingVideoDatapoint, MattingObject

from typing import Union

import cv2
# from utils_matting import

MAX_RETRIES = 100

class VideoMatteDataset(VisionDataset):
    def __init__(
        self,
        training: bool,
        mattingdataset: VideoMatteRawDataset,
        sampler: MattingSampler,
        multiplier: int,
        always_target=True,
        need_composition: bool=False,
        target_segments_available=True,
    ):
        assert isinstance(mattingdataset, VideoMatteRawDataset), "Dataset must be VideoMatteRawDataset"

        self.training = training
        self.video_dataset = mattingdataset
        self.sampler = sampler
        self.need_composition = need_composition

        self.repeat_factors = torch.ones(len(self.video_dataset), dtype=torch.float32)
        self.repeat_factors *= multiplier
        print(f"Raw dataset length = {len(self.video_dataset)}")

        self.curr_epoch = 0  # Used in case data loader behavior changes across epochs
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _get_datapoint(self, idx):

        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # sample a video
                video = self.video_dataset.get_video(idx)
                # sample frames and object indices to be used in a datapoint
                sampled_frms, bgrs = self.sampler.sample(video, self.curr_epoch)
                break  # Succesfully loaded video
            except Exception as e:
                if self.training:
                    logging.warning(
                        f"Loading failed (id={idx}); Retry {retry} with exception: {e}"
                    )
                    idx = random.randrange(0, len(self.video_dataset))
                else:
                    # Shouldn't fail to load a val video
                    raise e
        datapoint = self.construct(video, sampled_frms, bgrs)
        return datapoint


    def _composite_w_backgrounds(self, image, background, alpha, sizes):
        """
        Args:
        image : PIL RGB image
        background : PIL RGB image
        alpha : PIL L image
        Output:
        composite image : PIL RGB image
        """
        # resize the background to match the image size
        if background.size[0] < image.size[0] or background.size[1] < image.size[1]:
            scale = max(image.size[0] / background.size[0], image.size[1] / background.size[1])
            new_size = (int(background.size[0] * scale), int(background.size[1] * scale))
            background = background.resize(new_size, PILImage.LANCZOS)
        # center crop the background
        background = background.crop((
            (background.size[0] - image.size[0]) // 2,
            (background.size[1] - image.size[1]) // 2,
            (background.size[0] + image.size[0]) // 2,
            (background.size[1] + image.size[1]) // 2,
        ))
        return PILImage.composite(image, background, alpha)
    
    def _find_best_crop_region(self, img, alpha, size):
        min_x, min_y = 0, 0
        max_x, max_y = img.size[0], img.size[1]
        max_x = max(max_x - size[1], min_x + 1)
        max_y = max(max_y - size[0], min_y + 1)
        
        best_x, best_y = 0, 0
        best_foreground_pixels = 0
        
        for _ in range(5): 
            x, y = random.randint(min_x, max_x), random.randint(min_y, max_y)
            
            crop_alpha = alpha.crop((x, y, x + size[1], y + size[0]))
            
            foreground_pixels = (np.array(crop_alpha) > 127).sum()
            
            if foreground_pixels > best_foreground_pixels:
                best_x, best_y = x, y
                best_foreground_pixels = foreground_pixels
                
                if foreground_pixels > size[0] * size[1] * 0.10:  
                    break
        
        if best_foreground_pixels == 0:
            best_x = (max_x - min_x) // 2
            best_y = (max_y - min_y) // 2
        
        return best_x, best_y
    
    def _gen_transition_gt(self, alpha, masks=None, k_size=25, iterations=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (k_size, k_size))


        dilated = cv2.dilate(alpha[:, :].numpy(), kernel, iterations=iterations)
        eroded = cv2.erode(alpha[:, :].numpy(), kernel, iterations=iterations)
        trans_map = ((dilated - eroded) > 0).astype(float)

        # to tensor
        
        trans_map_tensor = torch.from_numpy(trans_map).float()
        return trans_map_tensor  

    def construct(self, video, sampled_frames, bgrs):
        """
        alphas are PIL image, not tensor
        """
        images = []
        rgb_images = load_images(sampled_frames)
        l_alphas = load_alphas(sampled_frames)
        size = (1024, 1024)
        reference_img = rgb_images[0]
        reference_alpha = l_alphas[0]
        
        crop_coordinates = self._find_best_crop_region(reference_img, reference_alpha, size)
        x, y = crop_coordinates
        
        resized_images = []
        resized_alphas = []
        for img, alpha in zip(rgb_images, l_alphas):
            if img.size[0] < x + size[1] or img.size[1] < y + size[0]:
                scale = max((x + size[1])/img.size[0], (y + size[0])/img.size[1])
                new_size = (int(img.size[0]*scale), int(img.size[1]*scale))
                img = img.resize(new_size, PILImage.LANCZOS)
                alpha = alpha.resize(new_size, PILImage.LANCZOS)
                    
            rs_img = img.crop((x, y, x + size[1], y + size[0]))
            rs_alpha = alpha.crop((x, y, x + size[1], y + size[0]))
            
            resized_alphas.append(rs_alpha)
            resized_images.append(rs_img)

            rgb_images = resized_images
            l_alphas = resized_alphas
        
        if self.need_composition:
            composed_rgb_images = [
                self._composite_w_backgrounds(img, bgr, alpha, size)
                for img, bgr, alpha in zip(rgb_images, bgrs, l_alphas)
            ]
            rgb_images = composed_rgb_images
        
        for frame_idx, frame in enumerate(sampled_frames):
            w, h = rgb_images[frame_idx].size
            image_i = rgb_images[frame_idx]
            alpha_i = l_alphas[frame_idx]            
            image_i_tensor = torch.from_numpy(np.array(image_i)).float().permute(2, 0, 1) / 255.0 # (3, 1024, 1024)
            alpha_i_tensor = torch.from_numpy(np.array(alpha_i)).float() / 255.0
            frame_i = Frame(
                data=image_i_tensor,
                data_alpha=alpha_i_tensor,
                objects=[],
            )

            alpha_tensor = alpha_i_tensor
            mask_tensor = (alpha_tensor > 0)

            
            fg_tensor = image_i_tensor / alpha_tensor 
            fg_tensor = torch.nan_to_num(fg_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            fg_tensor = torch.clamp(fg_tensor, 0, 1)
            
            trimap_tensor = self._gen_transition_gt(alpha_tensor)
            obj = MattingObject(
                object_id=0,  
                frame_index=frame_idx,
                segment=mask_tensor.bool(),  
                alphas=alpha_tensor,
                trimaps=trimap_tensor.bool(),
                fgs=fg_tensor
            ) 
            frame_i.objects.append(obj)
            images.append(frame_i)

        return MattingVideoDatapoint(
            frames=images,
            video_id=video.video_id,
            size=(h, w),
        )

    def __getitem__(self, idx):
        return self._get_datapoint(idx)

    def __len__(self):
        return len(self.video_dataset)


def load_images(frames):
    all_images = []
    cache = {}
    for frame in frames:
        if frame.data is None:
            # Load the frame rgb data from file
            path = frame.image_path
            if path in cache:
                all_images.append(deepcopy(all_images[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_images.append(PILImage.open(fopen).convert("RGB"))
            cache[path] = len(all_images) - 1
        else:
            # The frame rgb data has already been loaded
            # Convert it to a PILImage
            all_images.append(tensor_2_PIL_RGB(frame.data))

    return all_images

def load_alphas(frames):
    # recall that a frame contains alpha path
    all_alphas = []
    cache = {}
    for frame in frames:
        if frame.data_alpha is None:
            # Load the frame rgb data from file
            path = frame.alpha_path
            if path in cache:
                all_alphas.append(deepcopy(all_alphas[cache[path]]))
                continue
            with g_pathmgr.open(path, "rb") as fopen:
                all_alphas.append(PILImage.open(fopen).convert("L"))
            cache[path] = len(all_alphas) - 1
        else:
            # The frame alpha data has already been loaded
            # Convert it to a PILImage
            all_alphas.append(tensor_2_PIL_L(frame.data_alpha))

    return all_alphas

def tensor_2_PIL_RGB(data: torch.Tensor) -> PILImage.Image:
    data = data.cpu().numpy().transpose((1, 2, 0)) * 255.0
    data = data.astype(np.uint8)
    return PILImage.fromarray(data)

def tensor_2_PIL_L(data: torch.Tensor) -> PILImage.Image:
    """
    Input alpha tensor with shape (H, W), representing alpha channel
    """
    data = data.cpu().numpy() * 255.0
    data = data.astype(np.uint8)

    return PILImage.fromarray(data, mode='L')


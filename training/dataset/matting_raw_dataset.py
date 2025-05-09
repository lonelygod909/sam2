import os
import glob
import numpy as np
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig




@dataclass
class MattingFrame:
    image_path: str
    alpha_path: str
    trimap_path: Optional[str] = None
    mask_path: Optional[str] = None
    data: Optional[torch.Tensor] = None
    data_alpha: Optional[torch.Tensor] = None

@dataclass
class MattingVideo:
    video_name: str
    video_id: int
    frames: List[MattingFrame]

    def __len__(self):
        return len(self.frames)


class MattingRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class VideoMatteRawDataset(MattingRawDataset):
    def __init__(
        self,
        root,
        fgrs_folder,
        phas_folder,
        sample_rate=1,
        is_palette=True,
        truncate_video=-1,
    ):
        self.root = root
        self.fgrs_folder = os.path.join(root, fgrs_folder)
        self.phas_folder = os.path.join(root, phas_folder)
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.truncate_video = truncate_video
        
        self.video_names = sorted(os.listdir(self.fgrs_folder))


    def get_video(self, idx):
        video_name = self.video_names[idx]

        video_frame_root = os.path.join(self.fgrs_folder, video_name)

        video_alphas_root = os.path.join(self.phas_folder, video_name)
        
        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.png"))+
                            glob.glob(os.path.join(video_frame_root, "*.jpg")))
        all_alphas = sorted(glob.glob(os.path.join(video_alphas_root, "*.png"))+
                            glob.glob(os.path.join(video_alphas_root, "*.jpg")))

        if len(all_frames) == 0:
            raise ValueError(f"Video {video_name} has no frames.")
        
        if len(all_frames) != len(all_alphas):
            raise ValueError(f"Video {video_name} has different number of frames and alphas.")
        
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]

        frames = []
        for t, fpath in enumerate(all_frames[:: self.sample_rate]):
            # fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(MattingFrame(image_path=fpath, alpha_path=all_alphas[t]))
        video = MattingVideo(video_name, idx, frames)
        # return a video object which contains a list of frames which contains image_path, alpha_path.
        return video

    def __len__(self):
        return len(self.video_names)


import random
import os
from dataclasses import dataclass
from typing import List
from PIL import Image

MAX_RETRIES = 1000



class MattingSampler:
    def __init__(self, sort_frames=True):
        # frames are ordered by frame id when sort_frames is True
        self.sort_frames = sort_frames

    def sample(self, video):
        raise NotImplementedError()

    def _get_background(self):
        raise NotImplementedError()

class RandomUniformSampler(MattingSampler):
    def __init__(
        self,
        num_frames,
        reverse_time_prob=0.0,
        background_video_dir=None
    ):
        self.num_frames = num_frames
        self.reverse_time_prob = reverse_time_prob
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
        
    def _get_random_video_background(self, num_frames):
        """
        Get a random background sequence 
        """        
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - num_frames)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in range(num_frames):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = bgr.convert('RGB')
            bgrs.append(bgr)
        return bgrs    
    
    def sample(self, video, epoch=None):
        for retry in range(MAX_RETRIES):
            if len(video.frames) < self.num_frames:
                raise Exception(
                    f"Cannot sample {self.num_frames} frames from video {video.video_name} as it only has {len(video.frames)} annotated frames."
                )
            bgrs = self._get_random_video_background(self.num_frames)
            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = video.frames[start:start + self.num_frames]
            if random.uniform(0, 1) < self.reverse_time_prob:
                # Reverse time
                frames = frames[::-1]      
                bgrs = bgrs[::-1]           
            return frames, bgrs
        raise Exception(
            f"Cannot sample {self.num_frames} frames from video {video.video_name} after {MAX_RETRIES} retries."
        )

class EvalSampler(MattingSampler):
    """
    Matting Sampler for evaluation
    """

    def __init__(
        self,
        background_video_dir=None
    ):
        super().__init__()
        self.background_video_dir = background_video_dir
        self.background_video_clips = sorted(os.listdir(background_video_dir))
        self.background_video_frames = [sorted(os.listdir(os.path.join(background_video_dir, clip)))
                                        for clip in self.background_video_clips]
    
    def _get_random_video_background(self, num_frames):
        """
        Get a random background sequence 
        """        
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - num_frames)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in range(num_frames):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = bgr.convert('RGB')
            bgrs.append(bgr)
        return bgrs 

    def sample(self, video, epoch=None):
        """
        Sampling all the frames and all the objects
        """
        if self.sort_frames:
            # ordered by frame id
            frames = sorted(video.frames, key=lambda x: x.frame_idx)
        else:
            # use the original order
            frames = video.frames
        

        return frames, self._get_random_video_background(len(frames))
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from PIL import Image as PILImage
from tensordict import tensorclass

@tensorclass
class BatchedVideoMetaData:
    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedMattingImageMetaData:
    """
    This class represents metadata about a batch of videos.
    Attributes:
        unique_objects_identifier: A tensor of shape Bx3 containing unique identifiers for each object in the batch. Index consists of (video_id, obj_id, frame_id)
        frame_orig_size: A tensor of shape Bx2 containing the original size of each frame in the batch.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor

@tensorclass
class BatchedMattingVideoDatapoint:
    img_batch : torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    alphas: torch.FloatTensor
    trimaps: torch.BoolTensor
    fgs: torch.FloatTensor
    metadata: BatchedMattingImageMetaData

    dict_key: str

    def pin_memory(self, device=None):  
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """
        return self.img_batch.transpose(0, 1).flatten(0, 1)


@tensorclass
class BatchedMattingImageDatapoint:
    """
    This class represents a batch of videos with associated annotations and metadata.
    Attributes:
        img_batch: A [TxBxCxHxW] tensor containing the image data for each frame in the batch, where T is the number of frames per video, and B is the number of videos in the batch.
        obj_to_frame_idx: A [TxOx2] tensor containing the image_batch index which the object belongs to. O is the number of objects in the batch.
        masks: A [TxOxHxW] tensor containing binary masks for each object in the batch.
        alphas: A [TxOxHxW] tensor containing float alpha maps with value [0, 1] for each object in the batch.
        trimaps: A [TxOxHxW] tensor containing binary masks of unknown regions for each object in the batch.
        fgs: A [TxOxCxHxW] tensor containing the foreground data for each object in the batch.
        metadata: An instance of BatchedVideoMetaData containing metadata about the batch.
        dict_key: A string key used to identify the batch.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    alphas: torch.FloatTensor
    trimaps: torch.BoolTensor
    fgs: torch.FloatTensor
    metadata: BatchedMattingImageMetaData

    dict_key:str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """
        Returns the number of frames per video.
        """
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """
        Returns the number of videos in the batch.
        """
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """
        Returns a flattened tensor containing the object to img index.
        The flat index can be used to access a flattened img_batch of shape [(T*B)xCxHxW]
        """
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        flat_idx = video_idx * self.num_frames + frame_idx
        return flat_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """
        Returns a flattened img_batch_tensor of shape [(B*T)xCxHxW]
        """

        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class MattingObject:
    object_id: int
    frame_index: int
    segment: Union[torch.Tensor, dict]
    alphas: torch.Tensor
    trimaps: torch.Tensor
    fgs: torch.Tensor

@dataclass
class Frame:
    data: Union[torch.Tensor, PILImage.Image]
    objects: List[MattingObject]    
    # The image data for the frame, default is None
    data_alpha: Union[torch.Tensor, PILImage.Image]=None

@dataclass
class MattingImageDataPoint:
    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


@dataclass
class MattingVideoDatapoint:
    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[MattingImageDataPoint],
    dict_key,
) -> BatchedMattingImageDatapoint:
    """
    Args:
        batch: A list of VideoDatapoint instances.
        dict_key (str): A string key used to identify the batch.
    """
    img_batch = []
    for video in batch:
        img_batch += [torch.stack([frame.data for frame in video.frames], dim=0)]

    img_batch = torch.stack(img_batch, dim=0).permute((1, 0, 2, 3, 4))
    T = img_batch.shape[0]

    step_t_objects_identifier = [[] for _ in range(T)]
    step_t_frame_orig_size = [[] for _ in range(T)]

    step_t_masks = [[] for _ in range(T)]
    step_t_alphas = [[] for _ in range(T)]
    step_t_trimaps = [[] for _ in range(T)]
    step_t_fgs = [[] for _ in range(T)]
    step_t_obj_to_frame_idx = [
        [] for _ in range(T)
    ]  # List to store frame indices for each time step

    for video_idx, video in enumerate(batch):
        orig_video_id = video.video_id
        orig_frame_size = video.size
        for t, frame in enumerate(video.frames):
            objects = frame.objects
            for obj in objects:
                orig_obj_id = obj.object_id
                orig_frame_idx = obj.frame_index
                step_t_obj_to_frame_idx[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_t_masks[t].append(obj.segment.to(torch.bool))
                step_t_alphas[t].append(obj.alphas.to(torch.float))
                step_t_trimaps[t].append(obj.trimaps.to(torch.float))
                step_t_fgs[t].append(obj.fgs.to(torch.float))
                step_t_objects_identifier[t].append(
                    torch.tensor([orig_video_id, orig_obj_id, orig_frame_idx])
                )
                step_t_frame_orig_size[t].append(torch.tensor(orig_frame_size))
    
    obj_to_frame_idx = torch.stack(
        [
            torch.stack(obj_to_frame_idx, dim=0)
            for obj_to_frame_idx in step_t_obj_to_frame_idx
        ],
        dim=0,
    )
    masks = torch.stack([torch.stack(masks, dim=0) for masks in step_t_masks], dim=0)
    alphas = torch.stack([torch.stack(alphas, dim=0) for alphas in step_t_alphas], dim=0)
    trimaps = torch.stack([torch.stack(trimaps, dim=0) for trimaps in step_t_trimaps], dim=0)
    fgs = torch.stack([torch.stack(fgs, dim=0) for fgs in step_t_fgs], dim=0)
    objects_identifier = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_objects_identifier], dim=0
    )
    frame_orig_size = torch.stack(
        [torch.stack(id, dim=0) for id in step_t_frame_orig_size], dim=0
    )

    return BatchedMattingImageDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks,
        alphas=alphas,
        trimaps=trimaps,
        fgs = fgs,
        metadata=BatchedMattingImageMetaData(
            unique_objects_identifier=objects_identifier,
            frame_orig_size=frame_orig_size,
        ),
        dict_key=dict_key,
        batch_size=[T]
    )



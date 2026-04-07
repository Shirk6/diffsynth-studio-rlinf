
from __future__ import annotations

import imageio, os, torch, warnings, torchvision, argparse, json, ast
from ..utils import ModelConfig
from ..models.utils import load_state_dict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Tuple, Optional

from torch.utils.data import Dataset
from collections import OrderedDict
from pydantic import BaseModel, Field

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm", "gif"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        else:
            metadata = pd.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
            
    
    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image):
        if self.dynamic_resolution:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def _load_gif(self, file_path):
        gif_img = Image.open(file_path)
        frame_count = 0
        delays, frames = [], []
        while True:
            delay = gif_img.info.get('duration', 100) # ms
            delays.append(delay)
            rgb_frame = gif_img.convert("RGB")   
            croped_frame = self.crop_and_resize(rgb_frame, *self.get_height_width(rgb_frame))
            frames.append(croped_frame)             
            frame_count += 1
            try:
                gif_img.seek(frame_count)
            except:
                break
        # delays canbe used to calculate framerates
        # i guess it is better to sample images with stable interval,
        # and using minimal_interval as the interval, 
        # and framerate = 1000 / minimal_interval
        if any((delays[0] != i) for i in delays):
            minimal_interval = min([i for i in delays if i > 0])
            # make a ((start,end),frameid) struct
            start_end_idx_map = [((sum(delays[:i]), sum(delays[:i+1])), i) for i in range(len(delays))]
            _frames = []
            # according gemini-code-assist, make it more efficient to locate
            # where to sample the frame
            last_match = 0
            for i in range(sum(delays) // minimal_interval):
                current_time = minimal_interval * i
                for idx, ((start, end), frame_idx) in enumerate(start_end_idx_map[last_match:]):
                    if start <= current_time < end:
                        _frames.append(frames[frame_idx])
                        last_match = idx + last_match
                        break
            frames = _frames
        num_frames = len(frames)
        if num_frames > self.num_frames:
            num_frames = self.num_frames
        else:
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        frames = frames[:num_frames]
        return frames
    
    def load_video(self, file_path):
        if file_path.lower().endswith(".gif"):
            return self._load_gif(file_path)
        reader = imageio.get_reader(file_path)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame, *self.get_height_width(frame))
            frames.append(frame)
        reader.close()
        return frames
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image))
        frames = [image]
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                path = os.path.join(self.base_path, data[key])
                data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat

class RLinfNpyDataset(torch.utils.data.Dataset):
    """
    dataset_base_path/  
        rgb.npy  : [T, N, 3, H, W]
        traj.npy : [T, N, 3, H, W] 或 [T, N, ...]，需保证每帧能转成图片
    """

    def __init__(self, base_path='/opt/zsq/rlinf_dataset_1114_split/train_data', num_frames=9, repeat=1):
        self.base_path = base_path
        self.num_frames = num_frames
        self.repeat = repeat
        self.load_from_cache = False

        self.data_paths = []
        for step_name in sorted(os.listdir(base_path)):
            # step_path = os.path.join(base_path, step_name, "video/eval")
            step_path = os.path.join(base_path, step_name)
            if not os.path.isdir(step_path):
                continue
            for seed_name in sorted(os.listdir(step_path)):
                seed_path = os.path.join(step_path, seed_name)
                if os.path.isdir(seed_path):
                    self.data_paths.append(seed_path)
        if len(self.data_paths) == 0:
            raise ValueError(f"No valid data paths found under {base_path}")
        print(f'Found {len(self.data_paths)} data paths under {base_path}.')

        self.T_list = []
        self.N_list = []

        for p in self.data_paths:
            rgb_shape = np.load(os.path.join(p, "rgb.npy"), mmap_mode='r').shape
            T, N = rgb_shape[0], rgb_shape[1]
            self.T_list.append(T)
            self.N_list.append(N)

        self.total_env = sum(self.N_list)
        self.length = self.total_env * repeat 

        self.cum_N = np.cumsum(self.N_list)


    def __len__(self):
        return self.length
    def _locate_env(self, global_env_id):
        path_idx = np.searchsorted(self.cum_N, global_env_id, side='right')
        if path_idx == 0:
            env_id = global_env_id
        else:
            env_id = global_env_id - self.cum_N[path_idx - 1]
        return path_idx, env_id
    def __getitem__(self, idx):
        global_env_id = idx % self.total_env

        path_idx, env_id = self._locate_env(global_env_id)
        data_path = self.data_paths[path_idx]
        T = self.T_list[path_idx]
        rgb = np.load(os.path.join(data_path, "rgb.npy"), mmap_mode='r')
        actions = np.load(os.path.join(data_path, "actions.npy"), mmap_mode='r')


        if T > (self.num_frames-1):
            if np.random.rand() < 0.95:
                # 95% 随机采样
                if T >256:
                    start_idx = np.random.randint(0, 250)
                else:
                    start_idx = np.random.randint(0, T - self.num_frames + 2)

                consecutive_ids = np.arange(start_idx, start_idx + self.num_frames - 1)
                frame_ids = np.concatenate([[0], consecutive_ids])
            else:
                # 5% 使用硬编码 frame_ids
                frame_ids = np.array([0,0,0,0,0,1,2,3,4,5,6,7,8])
        else:
            raise ValueError(f"T={T} is too small for num_frames={self.num_frames}")
        
        video_np = rgb[frame_ids, env_id]  # shape [num_frames, 3, H, W]
        video_list = []
        for frame in video_np:
            if "/opt/zsq/rlinf_dataset_1114_split" in data_path:
                img = np.transpose(frame, (1, 2, 0))  # CHW → HWC
            else:
                img = frame
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255)
            video_list.append(Image.fromarray(img.astype(np.uint8)))

        # Action → Tensor
        action_np = actions[frame_ids, env_id]  # [num_frames, action_dim]
        action_tensor = torch.from_numpy(action_np).float()
        action_tensor[0] = torch.tensor([0., 0., 0., 0., 0., 0., -1.],
                                  dtype=action_tensor.dtype,
                                  device=action_tensor.device)
        return {
            "video": video_list,
            "reference_image": [video_list[0]],
            "action": action_tensor,
        }

class RLinfDataset(torch.utils.data.Dataset):
    """
    dataset_base_path/  
        rgb.npy  : [T, N, 3, H, W]
    """
    def __init__(
        self,
        base_path: Optional[list[str]] = None,
        Ta: int = 8,
        To: int = 4, # 4
        retain_reference_image: bool = True,
        retain_actions: bool = True,
        stride: int = 1,
        action_dim: int = 7,
        finish_step_shift: int = 440,
        repeat: int = 1,
    ):
        super().__init__()
        self.load_from_cache = False
        self.retain_reference_image = retain_reference_image
        self.retain_actions = retain_actions

        if not self.retain_reference_image:
            raise ValueError("retain_reference_image must be True; we use first image as reference image")

        self.Ta, self.To, self.stride = Ta, To, stride
        self.action_dim = action_dim
        self.finish_step_shift = finish_step_shift
        self.repeat = repeat

        # 扫描所有包含 rgb.npy 和 delta_actions.npy 的目录
        # 遵循与 MyNpyDatasetnew 相同的目录结构：base_path/step_name/video/eval/seed_name
        self.data_paths = []

        print(f"base_path: {base_path}")
        
        if isinstance(base_path, list):
            for base_path in base_path:
                for step_name in sorted(os.listdir(base_path)):
                    step_path = os.path.join(base_path, step_name)
                    if not os.path.isdir(step_path):
                        continue
                    for seed_name in sorted(os.listdir(step_path)):
                        seed_path = os.path.join(step_path, seed_name)
                        if os.path.isdir(seed_path):
                            self.data_paths.append(seed_path)
        else:
            for step_name in sorted(os.listdir(base_path)):
                # step_path = os.path.join(base_path, step_name, "video/eval")
                step_path = os.path.join(base_path, step_name)
                if not os.path.isdir(step_path):
                    continue
                for seed_name in sorted(os.listdir(step_path)):
                    seed_path = os.path.join(step_path, seed_name)
                    if os.path.isdir(seed_path):
                        self.data_paths.append(seed_path)
      
        if len(self.data_paths) == 0:
            raise FileNotFoundError(f"在 '{base_path}' 没找到任何包含 rgb.npy 和 delta_actions.npy 的目录")
        
        print(f'[RLinfDataset] Found {len(self.data_paths)} data paths under {base_path}.')

        # 预存每个 episode 的信息
        # 数据格式：rgb.npy 为 [T, N, 3, H, W]，delta_actions.npy 为 [T, N, action_dim]
        self.episode_info = []  # 每个元素是 (data_path, finish_step, T, N)
        self.sample_indices = []  # 每个元素是 (episode_idx, env_id, start)
        
        for data_path in self.data_paths:
            # 加载 video 和 action 的形状信息（不加载完整数据）
            video_shape = np.load(os.path.join(data_path, "rgb.npy"), mmap_mode='r').shape
            T, N = video_shape[0], video_shape[1]  # (T, N, C, H, W)

            if T > self.finish_step_shift:
                finish_step = self.finish_step_shift
            else:
                finish_step = T - 1
            
            episode_idx = len(self.episode_info)
            self.episode_info.append((data_path, finish_step, T, N))
            
            # 为每个环境和每个可能的窗口生成样本
            for env_id in range(N):
                for start in range(0, finish_step - self.Ta + 1, self.stride):
                    self.sample_indices.append((episode_idx, env_id, start))
        
        self.length = len(self.sample_indices) * repeat
        print(f"[RLinfDataset] 总共生成 {len(self.sample_indices)} 个样本，repeat={repeat}，总长度={self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 计算实际的样本索引
        sample_idx = idx % len(self.sample_indices)
        episode_idx, env_id, start = self.sample_indices[sample_idx]
        data_path, finish_step, T, N = self.episode_info[episode_idx]
        # 懒加载当前 npy 文件
        video_np = np.load(os.path.join(data_path, "rgb.npy"), mmap_mode='r')  # (T, N, C, H, W)
        # action_np = np.load(os.path.join(data_path, "delta_actions.npy"), mmap_mode='r')  # (T, N, action_dim)
        action_np = np.load(os.path.join(data_path, "actions.npy"), mmap_mode='r')

        vs, ve = start - self.To + 1, start + self.Ta + 1

        if not self.retain_actions:
            action_s, action_e = start + 1, start + self.Ta + 1
        else:
            action_s, action_e = start - self.To + 1, start + self.Ta + 1

        if len(video_np) < ve:
            raise ValueError(f"video length is not right: {len(video_np)} < {ve}")
        
        if vs < 0:
            # 直接将第一帧 repeat To 次模拟 padding
            pad = np.repeat(video_np[0:1, env_id], self.To, axis=0)  # [1, C, H, W] -> [-vs, C, H, W]
            vid_win = np.concatenate([pad, video_np[:self.Ta, env_id]], axis=0)  # [ve-vs, C, H, W]
            # 对于 video_np 进行 repeat
            # pad = np.repeat(video_np[0:1, env_id], -vs, axis=0)  # [1, C, H, W] -> [-vs, C, H, W]
            # vid_win = np.concatenate([pad, video_np[:ve, env_id]], axis=0)  # [ve-vs, C, H, W]
        else:
            vid_win = video_np[vs:ve, env_id]  # [ve-vs, C, H, W]

        if self.retain_reference_image:
            vid_win = np.concatenate([video_np[0:1, env_id], vid_win], axis=0)  # [1+ve-vs, C, H, W]
        
        if action_s < 0:
            # 将第一帧的 action repeat To 次模拟 padding
            pad_action = np.repeat(action_np[0:1, env_id], self.To, axis=0)  # [1, action_dim] -> [-action_s, action_dim]
            act_win = np.concatenate([pad_action, action_np[:self.Ta, env_id]], axis=0)  # [action_e-action_s, action_dim]
        else:
            act_win = action_np[action_s:action_e, env_id]  # [action_e-action_s, action_dim]]

        if self.retain_reference_image and self.retain_actions:
            act_win = np.concatenate([action_np[0:1, env_id], act_win], axis=0)  # [1+action_e-action_s, action_dim]

        action_tensor = torch.from_numpy(act_win).float()
        
        # 确保 vid_win 的形状正确
        assert vid_win.shape[0] == self.Ta + self.To + 1, f"video len is not right: {vid_win.shape[0]} != {self.Ta + self.To}"      
        # 确保 act_win 的长度正确（应该是 Ta + To）
        if not self.retain_actions:
            assert act_win.shape[0] == self.Ta, f"action len is not right: {act_win.shape[0]} != {self.Ta}"
        else:
            assert act_win.shape[0] == self.Ta + self.To + 1, f"action len is not right: {act_win.shape[0]} != {self.Ta + self.To}"
        
        # 转换为与 MyNpyDatasetnew 对齐的格式
        # video: List[PIL.Image]，每个图像是 uint8
        video_list = []
        for frame in vid_win:
            # frame shape: (C, H, W) -> 转换为 (H, W, C)
            if frame.shape[0] == 3:  # CHW 格式
                frame = frame.transpose(1, 2, 0)  # CHW → HWC
            # 确保是 uint8
            if frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255)
            img = frame.astype(np.uint8)
            video_list.append(Image.fromarray(img))

        return {
            "video": video_list,
            "reference_image": [video_list[0]],
            "action": action_tensor,
        }


class SimpleVLARealWorldRLinfDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path: str | list[str] | tuple[str, ...],
        Ta: int = 8,
        To: int = 4, # 4
        reference_image: bool = True,
        stride: int = 1,
        action_dim: int = 7,
        finish_step_shift: int = 0,
        repeat: int = 1,
        retain_action=False
    ):
        super().__init__()
        self.load_from_cache = False
        self.reference_image = reference_image
        self.retain_action = retain_action      # action is 

        assert not self.retain_action, "retain_action暂时不支持"

        if not self.reference_image:
            To = To + 1 # 不包含 reference_image 时，To 需要 +1, 即使用五帧

        self.Ta, self.To, self.stride = Ta, To, stride
        self.context_length = self.Ta + self.To
        self.action_dim = action_dim
        self.finish_step_shift = finish_step_shift
        self.repeat = repeat

        # 扫描 base_path 下的所有 .npy 文件
        # 每个 .npy 文件包含一个轨迹，格式为：数组[(T,)] 其中每个元素是 dict{'observations': (H,W,C), 'actions': (action_dim,)}
        self.data_paths = []
        
        base_paths = list(base_path) if isinstance(base_path, (list, tuple)) else [base_path]
        for current_base_path in base_paths:
            for filename in sorted(os.listdir(current_base_path)):
                if filename.endswith('.npy'):
                    file_path = os.path.join(current_base_path, filename)
                    self.data_paths.append(file_path)
      
        if len(self.data_paths) == 0:
            raise FileNotFoundError(f"在 '{base_paths}' 没找到任何 .npy 文件")
        
        print(f'[SimpleVLARealWorldRLinfDataset] Found {len(self.data_paths)} trajectory files under {base_paths}.')

        # 预存每个 episode 的信息
        # 数据格式：每个 .npy 文件是 [T] 个 dict，每个 dict 包含 'observations' 和 'actions'
        self.episode_info = []  # 每个元素是 (file_path, finish_step, T)
        self.sample_indices = []  # 每个元素是 (episode_idx, start)
        
        for file_path in self.data_paths:
            # 加载轨迹的长度信息（包含 Python 对象的数组不能使用 mmap）
            traj_data = np.load(file_path, allow_pickle=True)
            T = len(traj_data)  # 轨迹长度
            finish_step = T - 1
            
            episode_idx = len(self.episode_info)
            self.episode_info.append((file_path, finish_step, T))
            
            # 为每个可能的窗口生成样本
            for start in range(0, finish_step - self.Ta + 1, self.stride):
                self.sample_indices.append((episode_idx, start))
        
        self.length = len(self.sample_indices) * repeat
        print(f"[SimpleVLARealWorldRLinfDataset] 总共生成 {len(self.sample_indices)} 个样本，repeat={repeat}，总长度={self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 计算实际的样本索引
        sample_idx = idx % len(self.sample_indices)
        episode_idx, start = self.sample_indices[sample_idx]
        file_path, finish_step, T = self.episode_info[episode_idx]
        
        # 懒加载当前轨迹文件（包含 Python 对象的数组不能使用 mmap）
        traj_data = np.load(file_path, allow_pickle=True)  # [T] 个 dict

        # 采样逻辑：遵循 SimpleVLARLinfDataset 的采样逻辑
        vs, ve = start - self.To + 1, start + self.Ta + 1
        # action_s, action_e = start + 1, start + self.Ta + 1
        if not self.retain_action:
            action_s, action_e = start, start + self.Ta
        else:
            action_s, action_e = start - self.To, start + self.Ta

        if len(traj_data) < ve:
            raise ValueError(f"trajectory length is not right: {len(traj_data)} < {ve}")
        
        # 提取视频帧和动作
        if vs < 0:
            # 直接使用前面的帧进行 padding
            pad_frames = [traj_data[0]['observations'] for _ in range(self.To)]
            vid_frames = [traj_data[i]['observations'] for i in range(self.Ta)]
            vid_win = pad_frames + vid_frames
        else:
            vid_win = [traj_data[i]['observations'] for i in range(vs, ve)]

        if self.reference_image:
            # 添加第0帧作为参考图像
            vid_win = [traj_data[0]['observations']] + vid_win
        
        # 提取动作
        # if self.retain_action and action_s < -1:
        #     pad_actions = np.zeros((self.To+1, self.action_dim))       # pad 0动作
        #     act_frames = [traj_data[i]['actions'] for i in range(self.Ta-1)]
        #     act_win = pad_actions + act_frames
        # elif self.retain_action and action_s == -1:
        #     pad_actions = np.zeros((1, self.action_dim))               # pad 0 动作
        #     act_frames = [traj_data[i]['actions'] for i in range(self.Ta + self.To - 1)]
        #     act_win = pad_actions + act_frames
        # else:
        #     act_win = np.array([traj_data[i]['actions'] for i in range(action_s, action_e)])  # [Ta, action_dim]

        act_win = np.array([traj_data[i]['actions'] for i in range(action_s, action_e)])
        action_tensor = torch.from_numpy(act_win).float()
        
        # 确保长度正确
        if self.reference_image:
            expected_len = self.Ta + self.To + 1
            assert len(vid_win) == expected_len, f"video len is not right: {len(vid_win)} != {expected_len}"
        else:
            expected_len = self.Ta + self.To
            assert len(vid_win) == expected_len, f"video len is not right: {len(vid_win)} != {expected_len}"
        assert act_win.shape[0] == self.Ta, f"action len is not right: {act_win.shape[0]} != {self.Ta}"
        
        # 转换为与 MyNpyDatasetnew 对齐的格式
        # video: List[PIL.Image]，每个图像是 uint8
        video_list = []
        for frame in vid_win:
            # frame shape: 应该已经是 (H, W, C)
            # 确保是 uint8
            if frame.max() <= 1.0:
                frame = (frame * 255).clip(0, 255)
            img = frame.astype(np.uint8)
            video_list.append(Image.fromarray(img))
        
        if vs >= 0:
            frame_ids = np.arange(vs, ve)
        else:
            # 前面 pad 的部分用 0 表示
            frame_ids = np.concatenate([np.zeros(-vs, dtype=np.int32), np.arange(0, ve)])

        return {
            "video": video_list,
            "prompt": "机械臂根据控制轨迹进行相应的移动",
            "reference_image": [video_list[0]],
            "action": action_tensor,
            "idx": frame_ids,
        }

class RLinfLeRobotObsDataset(torch.utils.data.Dataset):
    """
    A compact LeRobot wrapper aligned with RLinfDataset output format.
    Only observation videos are loaded; actions are zero placeholders.
    """

    DEFAULT_VIDEO_KEY_MAP = {
        "AgiBot_merge": "observation.images.head",
        "droid_1.0.1": "observation.images.exterior_1_left",
        "robomind_agilex_3rgb": "observation.images.camera_front",
        "robomind_franka_1rgb": "observation.images.camera_top",
    }
    DEFAULT_ACTION_KEY_MAP = {
        "AgiBot_merge": "actions.delta",
        "droid_1.0.1": "action.original",
        # Robomind franka action is composed by concatenating the three keys below.
        "robomind_franka_1rgb": (
            "actions.eef_pos",
            "actions.eef_rot",
            "actions.eef_gripper",
        ),
    }

    def __init__(
        self,
        base_path: Optional[list[str]] = None,
        Ta: int = 8,
        To: int = 4,
        retain_reference_image: bool = True,
        retain_actions: bool = False,
        stride: int = 1,
        action_dim: int = 14,
        image_size: Tuple[int, int] = (256, 256),
        finish_step_shift: int = 0,
        repeat: int = 1,
        video_key_map: Optional[dict[str, str]] = None,
        max_open_video_readers: int = 8,
    ):
        super().__init__()
        self.load_from_cache = False
        self.retain_reference_image = retain_reference_image
        self.retain_actions = retain_actions
        if not self.retain_reference_image:
            raise ValueError("retain_reference_image must be True")

        self.Ta, self.To, self.stride = Ta, To, stride
        self.image_size = image_size
        # Unified action dim for mixed datasets.
        self.action_dim = action_dim
        self.finish_step_shift = finish_step_shift
        self.repeat = repeat

        self.max_open_video_readers = max(1, int(max_open_video_readers))
        self._video_readers = OrderedDict()
        self._action_cache = OrderedDict()
        self.max_open_action_tables = 16

        if base_path is None:
            cwd = os.getcwd()
            base_path = [
                os.path.join(cwd, "robomind_franka_1rgb"),
                os.path.join(cwd, "droid_1.0.1"),
                os.path.join(cwd, "AgiBot_merge"),
                os.path.join(cwd, "robomind_agilex_3rgb"),
            ]
        if isinstance(base_path, str):
            base_path = [base_path]

        key_map = dict(self.DEFAULT_VIDEO_KEY_MAP)
        if video_key_map is not None:
            key_map.update(video_key_map)

        self.episode_info = []
        self.sample_indices = []  # (episode_id, start)

        for root in base_path:
            root = os.path.abspath(root)
            dataset_name = os.path.basename(root.rstrip("/"))
            info_json_path = os.path.join(root, "meta", "info.json")
            if not os.path.exists(info_json_path):
                raise FileNotFoundError(f"Missing meta/info.json: {info_json_path}")

            with open(info_json_path, "r", encoding="utf-8") as f:
                info_meta = json.load(f)

            video_key = key_map.get(dataset_name)
            if video_key is None:
                raise KeyError(f"Missing video_key mapping for `{dataset_name}`")
            features = info_meta.get("features", {})
            if video_key not in features:
                raise KeyError(f"Video key `{video_key}` not found in {info_json_path}")
            action_key_spec = self.DEFAULT_ACTION_KEY_MAP.get(dataset_name, None)
            if action_key_spec is not None:
                if isinstance(action_key_spec, tuple):
                    for key in action_key_spec:
                        if key not in features:
                            raise KeyError(f"Action key `{key}` not found in {info_json_path}")
                else:
                    if action_key_spec not in features:
                        raise KeyError(f"Action key `{action_key_spec}` not found in {info_json_path}")

            episodes_root = os.path.join(root, "meta", "episodes")
            if not os.path.isdir(episodes_root):
                raise FileNotFoundError(f"Missing meta/episodes: {episodes_root}")

            for chunk_name in sorted(os.listdir(episodes_root)):
                chunk_path = os.path.join(episodes_root, chunk_name)
                if not os.path.isdir(chunk_path):
                    continue
                for file_name in sorted(os.listdir(chunk_path)):
                    if not file_name.endswith(".parquet"):
                        continue
                    episode_meta_file = os.path.join(chunk_path, file_name)
                    ep_df = pd.read_parquet(episode_meta_file)

                    video_chunk_col = f"videos/{video_key}/chunk_index"
                    video_file_col = f"videos/{video_key}/file_index"

                    for _, row in ep_df.iterrows():
                        length = int(row["length"])
                        if length <= self.Ta:
                            continue

                        if video_chunk_col in ep_df.columns and video_file_col in ep_df.columns:
                            video_chunk_index = int(row[video_chunk_col])
                            video_file_index = int(row[video_file_col])
                        else:
                            video_chunk_index = int(row["data/chunk_index"])
                            video_file_index = int(row["data/file_index"])

                        video_path = os.path.join(
                            root,
                            "videos",
                            video_key,
                            f"chunk-{video_chunk_index:03d}",
                            f"file-{video_file_index:03d}.mp4",
                        )
                        if not os.path.exists(video_path):
                            continue

                        episode_id = len(self.episode_info)
                        self.episode_info.append(
                            {
                                "dataset_name": dataset_name,
                                "video_path": video_path,
                                "length": length,
                                "data_chunk_index": int(row["data/chunk_index"]),
                                "data_file_index": int(row["data/file_index"]),
                                "dataset_from_index": int(row["dataset_from_index"]),
                                "dataset_to_index": int(row["dataset_to_index"]),
                                "data_path": os.path.join(
                                    root,
                                    "data",
                                    f"chunk-{int(row['data/chunk_index']):03d}",
                                    f"file-{int(row['data/file_index']):03d}.parquet",
                                ),
                            }
                        )

                        finish_step = length - 1
                        for start in range(0, finish_step - self.Ta + 1, self.stride):
                            self.sample_indices.append((episode_id, start))

        if len(self.sample_indices) == 0:
            raise RuntimeError("No valid samples found from the provided datasets")

        self.length = len(self.sample_indices) * self.repeat
        print(
            f"[RLinfLeRobotObsDataset] episodes={len(self.episode_info)} "
            f"samples={len(self.sample_indices)} repeat={self.repeat} total={self.length}"
        )

    def __len__(self):
        return self.length

    def _get_video_reader(self, video_path: str):
        if video_path in self._video_readers:
            reader = self._video_readers.pop(video_path)
            self._video_readers[video_path] = reader
            return reader

        if len(self._video_readers) >= self.max_open_video_readers:
            _, old_reader = self._video_readers.popitem(last=False)
            try:
                old_reader.close()
            except Exception:
                pass

        reader = imageio.get_reader(video_path, format="ffmpeg")
        self._video_readers[video_path] = reader
        return reader

    def _load_frame(self, video_path: str, frame_idx: int) -> np.ndarray:
        reader = self._get_video_reader(video_path)
        return reader.get_data(int(frame_idx))

    def _get_action_file_df(self, data_path: str, columns: list[str]) -> pd.DataFrame:
        if data_path in self._action_cache:
            cached_df = self._action_cache.pop(data_path)
            self._action_cache[data_path] = cached_df
            return cached_df

        if len(self._action_cache) >= self.max_open_action_tables:
            self._action_cache.popitem(last=False)

        df = pd.read_parquet(data_path, columns=columns)
        self._action_cache[data_path] = df
        return df

    def _get_episode_action_array(self, ep: dict) -> np.ndarray:
        dataset_name = ep["dataset_name"]
        action_key_spec = self.DEFAULT_ACTION_KEY_MAP.get(dataset_name, None)
        length = ep["length"]
        if action_key_spec is None:
            return np.zeros((length, self.action_dim), dtype=np.float32)

        if isinstance(action_key_spec, tuple):
            columns = list(action_key_spec)
        else:
            columns = [action_key_spec]

        df = self._get_action_file_df(ep["data_path"], columns)
        start = ep["dataset_from_index"]
        end = ep["dataset_to_index"]
        if end - start != length:
            end = start + length
        # Some datasets store global indices in metadata. Clamp to local parquet range.
        start = max(0, min(int(start), len(df)))
        end = max(start, min(int(end), len(df)))

        action_parts = []
        for key in columns:
            values = df[key].iloc[start:end].to_numpy()
            if len(values) == 0:
                arr = np.zeros((0, 1), dtype=np.float32)
            else:
                first_value = values[0]
                if np.isscalar(first_value):
                    arr = np.asarray(values, dtype=np.float32).reshape(-1, 1)
                else:
                    arr = np.stack(values, axis=0).astype(np.float32)
                    if arr.ndim == 1:
                        arr = arr[:, None]
            action_parts.append(arr)

        action = np.concatenate(action_parts, axis=1) if len(action_parts) > 0 else np.zeros((0, self.action_dim), dtype=np.float32)
        if action.shape[0] < length:
            pad = np.zeros((length - action.shape[0], action.shape[1]), dtype=np.float32)
            action = np.concatenate([action, pad], axis=0)
        elif action.shape[0] > length:
            action = action[:length]
        if action.shape[1] < self.action_dim:
            pad = np.zeros((action.shape[0], self.action_dim - action.shape[1]), dtype=np.float32)
            action = np.concatenate([action, pad], axis=1)
        elif action.shape[1] > self.action_dim:
            raise ValueError(f"Action dimension mismatch: {action.shape[1]} > {self.action_dim}")
        return action

    def __getitem__(self, idx):
        sample_idx = idx % len(self.sample_indices)
        episode_id, start = self.sample_indices[sample_idx]
        ep = self.episode_info[episode_id]
        video_path = ep["video_path"]
        action_np = self._get_episode_action_array(ep)

        vs, ve = start - self.To + 1, start + self.Ta + 1
        ref_frame = self._load_frame(video_path, 0)

        frames = []
        if vs < 0:
            for _ in range(-vs):
                frames.append(ref_frame)
            begin = 0
        else:
            begin = vs

        for frame_idx in range(begin, ve):
            frames.append(self._load_frame(video_path, frame_idx))

        target_hw = (self.image_size[1], self.image_size[0]) if self.image_size is not None else None
        video_list = []
        for frame in frames:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)
            img = Image.fromarray(frame)
            if target_hw is not None:
                img = img.resize(target_hw, Image.BILINEAR)
            video_list.append(img)

        if self.retain_reference_image:
            ref_img = Image.fromarray(ref_frame.astype(np.uint8))
            if target_hw is not None:
                ref_img = ref_img.resize(target_hw, Image.BILINEAR)
            video_list = [ref_img] + video_list

        expected_video_len = self.Ta + self.To + 1
        assert len(video_list) == expected_video_len, (
            f"video len is not right: {len(video_list)} != {expected_video_len}"
        )

        if not self.retain_actions:
            action_s, action_e = start + 1, start + self.Ta + 1
        else:
            action_s, action_e = start - self.To + 1, start + self.Ta + 1

        if action_s < 0:
            pad_action = np.repeat(action_np[0:1], -action_s, axis=0)
            act_win = np.concatenate([pad_action, action_np[:action_e]], axis=0)
        else:
            act_win = action_np[action_s:action_e]

        if self.retain_reference_image and self.retain_actions:
            act_win = np.concatenate([action_np[0:1], act_win], axis=0)

        action_tensor = torch.from_numpy(act_win.astype(np.float32))

        return {
            "video": video_list,
            "reference_image": [video_list[0]],
            "action": action_tensor,
        }

    def __del__(self):
        for _, reader in getattr(self, "_video_readers", {}).items():
            try:
                reader.close()
            except Exception:
                pass


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, upcast_dtype=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        if upcast_dtype is not None:
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.to(upcast_dtype)
        return model


    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict
    
    
    def transfer_data_to_device(self, data, device, torch_float_dtype=None):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
                if torch_float_dtype is not None and data[key].dtype in [torch.float, torch.float16, torch.bfloat16]:
                    data[key] = data[key].to(torch_float_dtype)
        return data
    
    
    def parse_model_configs(self, model_paths, model_id_with_origin_paths, enable_fp8_training=False):
        offload_dtype = torch.float8_e4m3fn if enable_fp8_training else None
        model_configs = []
        print(f"model_paths: {model_paths}")
        if model_paths is not None:
            model_paths = json.loads(model_paths)
            model_configs += [ModelConfig(path=path, offload_dtype=offload_dtype) for path in model_paths]
        if model_id_with_origin_paths is not None:
            model_id_with_origin_paths = model_id_with_origin_paths.split(",")
            model_configs += [ModelConfig(model_id=i.split(":")[0], origin_file_pattern=i.split(":")[1], offload_dtype=offload_dtype) for i in model_id_with_origin_paths]
        return model_configs
    
    
    def switch_pipe_to_training_mode(
        self,
        pipe,
        trainable_models,
        lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=None,
        enable_fp8_training=False,
    ):
        # Scheduler
        pipe.scheduler.set_timesteps(1000, training=True)
        # ❗
        # pipe.scheduler.set_timesteps(1000, training=True, enhance5steps=True)
        
        # Freeze untrainable models
        pipe.freeze_except([] if trainable_models is None else trainable_models.split(","))
        
        # Enable FP8 if pipeline supports
        if enable_fp8_training and hasattr(pipe, "_enable_fp8_lora_training"):
            pipe._enable_fp8_lora_training(torch.float8_e4m3fn)
        
        # Add LoRA to the base models
        if lora_base_model is not None:
            model = self.add_lora_to_model(
                getattr(pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                lora_rank=lora_rank,
                upcast_dtype=pipe.torch_dtype,
            )
            if lora_checkpoint is not None:
                state_dict = load_state_dict(lora_checkpoint)
                state_dict = self.mapping_lora_state_dict(state_dict)
                load_result = model.load_state_dict(state_dict, strict=False)
                print(f"LoRA checkpoint loaded: {lora_checkpoint}, total {len(state_dict)} keys")
                if len(load_result[1]) > 0:
                    print(f"Warning, LoRA key mismatch! Unexpected keys in LoRA checkpoint: {load_result[1]}")
            setattr(pipe, lora_base_model, model)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    save_epochs: int = 1,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        save_epochs = args.save_epochs
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        ###验证集
        val_interval = args.val_interval # 5
    writer = SummaryWriter(log_dir=os.path.join(args.output_path, "tensorboard") if args is not None else "./tensorboard")

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, val_dataloader, scheduler)

    global_step = 0
    epoch_loss = 0.0
    epoch_steps = 0

    for epoch_id in range(num_epochs):

        epoch_loss = 0.0
        epoch_steps = 0

        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                print(f'Epoch {epoch_id} Step {global_step} Loss: {loss.item()}')
                accelerator.backward(loss)

                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()

                if accelerator.is_main_process:
                    writer.add_scalar("Loss/step", loss.item(), global_step)
                epoch_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # to avoid the too long epoch 
                if epoch_steps > 500:
                    break
        if accelerator.is_main_process and epoch_steps > 0:
            writer.add_scalar("Loss/epoch", epoch_loss / epoch_steps, epoch_id)

        if val_dataloader is not None and (epoch_id + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            if accelerator.is_main_process:
                print(f"\nRunning validation for epoch {epoch_id}...")

            for data in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    if getattr(val_dataset, 'load_from_cache', False):
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    

                    avg_loss = accelerator.gather(loss).mean().item()
                    val_loss += avg_loss
                    val_steps += 1

                    if val_steps > 2001:
                        break
            
            if val_steps > 0:
                avg_val_loss = val_loss / val_steps
                if accelerator.is_main_process:
                    writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch_id)
                    print(f"Epoch {epoch_id} Validation Loss: {avg_val_loss}")
            
            model.train() 
            # --------------------
        if save_steps is None and (epoch_id + 1) % save_epochs == 0:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)
    writer.close()


def launch_data_process_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in tqdm(enumerate(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data, return_inputs=True)
                torch.save(data, save_path)



def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--audio_processor_config", type=str, default=None, help="Model ID with origin paths to the audio processor config, e.g., Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--save_epochs", type=int, default=1, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    
    return parser



def flux_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--align_to_opensource_format", default=False, action="store_true", help="Whether to align the lora format to opensource format. Only for DiT's LoRA.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    return parser



def qwen_image_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", required=True, help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1024*1024, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--data_file_keys", type=str, default="image", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Paths to tokenizer.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--processor_path", type=str, default=None, help="Path to the processor. If provided, the processor will be used for image editing.")
    parser.add_argument("--enable_fp8_training", default=False, action="store_true", help="Whether to enable FP8 training. Only available for LoRA training on a single GPU.")
    parser.add_argument("--task", type=str, default="sft", required=False, help="Task type.")
    return parser




# python -m diffsynth.trainers.utils
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="RLinfDataset", choices=["RLinfDataset", "RLinfLeRobotObsDataset"])
    parser.add_argument("--base_path", type=str, default='/mnt/project_rlinf/jzn/dataset/simulation/dataset_for_posttrain_worldmodel_libero_spatial/base_policy_rollout/train_data')
    # parser.add_argument("--base_path", type=str, default='["/mnt/project_rlinf/jlchen/datasets/robomind_franka_1rgb","/mnt/project_rlinf/jlchen/datasets/droid_1.0.1","/mnt/project_rlinf/jlchen/datasets/AgiBot_merge"]')
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    base_path = args.base_path

    if args.dataset_type == "RLinfLeRobotObsDataset":
        dataset = RLinfLeRobotObsDataset(base_path=base_path, repeat=args.repeat, action_dim=args.action_dim)
    elif args.dataset_type == "RLinfNpyDataset":
        dataset = RLinfNpyDataset(base_path=base_path, repeat=args.repeat, num_frames=13)
    elif args.dataset_type == "RLinfDataset":
        dataset = RLinfDataset(base_path=base_path, retain_actions=True)

    print(f"dataset_type={args.dataset_type} len={len(dataset)}")
    probe_indices = [0, 1, 2, 10, 100][: args.num_samples]
    for i in probe_indices:
        sample = dataset[i]
        action = sample['action']
        print(
            f"idx={i} video_len={len(sample['video'])} action_shape={tuple(sample['action'].shape)} "
            f"ref_len={len(sample['reference_image'])}"
        )
        import pdb; pdb.set_trace()
        print(sample['action'])

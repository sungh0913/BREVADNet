import os
import glob
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])

class SlidingWindowDataset(Dataset):
    def __init__(self, frame_dir, clip_len=5, transform_fn=None, stride=1, recursive=False, cache_to_ram=False):
        self.frame_dir = frame_dir
        self.clip_len = clip_len
        self.stride = stride
        self.image_paths = []
        self.video_map = {}
        self.cache_to_ram = cache_to_ram
        self.frames_cache = {} 

        def get_frame_number(filename):
            matches = re.findall(r'\d+', filename)
            if matches:
                return int(matches[-1])
            return -1

        def is_valid_file(filename):
            return (not filename.startswith('.') and 
                    '.ipynb_checkpoints' not in filename)

        def is_valid_dir(dirname):
            return (not dirname.startswith('.') and 
                    '.ipynb_checkpoints' not in dirname)

        if recursive:
            video_subdirs = sorted([d.path for d in os.scandir(frame_dir) 
                                  if d.is_dir() and is_valid_dir(d.name)])
            
            for video_dir in video_subdirs:
                video_name = os.path.basename(video_dir)
                jpg_files = glob.glob(os.path.join(video_dir, '*.jpg'))
                
                if jpg_files:
                    valid_files = [f for f in jpg_files if is_valid_file(os.path.basename(f))]
                    if not valid_files: continue
                    
                    try:
                        jpg_files_sorted = sorted(valid_files, key=lambda x: get_frame_number(os.path.basename(x)))
                    except Exception as e:
                        print(f"Sorting error in {video_name}: {e}")
                        continue

                    self.image_paths.append(jpg_files_sorted)
                    self.video_map[video_name] = jpg_files_sorted
        else:
            video_name = os.path.basename(frame_dir)
            jpg_files = glob.glob(os.path.join(frame_dir, '*.jpg'))
            if jpg_files:
                valid_files = [f for f in jpg_files if is_valid_file(os.path.basename(f))]
                if valid_files:
                    try:
                        jpg_files_sorted = sorted(valid_files, key=lambda x: get_frame_number(os.path.basename(x)))
                        self.image_paths.append(jpg_files_sorted)
                        self.video_map[video_name] = jpg_files_sorted
                    except Exception as e:
                        print(f"Sorting error in {video_name}: {e}")

        self.folder_samples = []
        self.cumulative_samples = [0]
        for folder_paths in self.image_paths:
            folder_len = max(0, (len(folder_paths) - self.clip_len) // self.stride + 1)
            self.folder_samples.append(folder_len)
            self.cumulative_samples.append(self.cumulative_samples[-1] + folder_len)

        self.length = self.cumulative_samples[-1]

        im_resize = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        self.transform_fn = transforms.Compose([im_resize, transforms.ToTensor(), normalize])

        if self.cache_to_ram:
            all_paths_flat = [p for folder in self.image_paths for p in folder]
            print(f"[Dataset] Loading {len(all_paths_flat)} frames into RAM cache...")
            
            for img_path in tqdm(all_paths_flat, desc="Caching Images"):
                if img_path not in self.frames_cache:
                    try:
                        with Image.open(img_path) as img:
                            image = img.convert("RGB")
                            self.frames_cache[img_path] = self.transform_fn(image)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
            print("[Dataset] RAM Cache initialization complete.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        folder_idx = 0
        for i in range(1, len(self.cumulative_samples)):
            if index < self.cumulative_samples[i]:
                folder_idx = i - 1
                break

        local_idx = index - self.cumulative_samples[folder_idx]
        start_idx = local_idx * self.stride

        if not self.image_paths: return None

        folder_paths = self.image_paths[folder_idx]

        if start_idx + self.clip_len > len(folder_paths):
            return None

        clip_paths = folder_paths[start_idx:start_idx + self.clip_len]

        clip_frames = []
        
        if self.cache_to_ram:
            for img_path in clip_paths:
                if img_path in self.frames_cache:
                    clip_frames.append(self.frames_cache[img_path])
                else:
                    with Image.open(img_path) as img:
                        image = img.convert("RGB")
                        clip_frames.append(self.transform_fn(image))
        else:
            for img_path in clip_paths:
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
                    image = self.transform_fn(image)
                    clip_frames.append(image)

        history_sequence = torch.stack(clip_frames[:-1], dim=0)
        future_frame = clip_frames[-1]
        full_sequence = torch.stack(clip_frames, dim=0)
        folder_name = os.path.basename(os.path.dirname(clip_paths[0]))

        return {
            'video_name': folder_name,
            'start_idx': start_idx,
            'history_sequence': history_sequence,
            'future_frame': future_frame,
            'full_sequence': full_sequence
        }
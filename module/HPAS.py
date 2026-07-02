import torch
import torch.nn.functional as F
import random
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_frame_as_tensor(frame_path, H, W, device):
    try:
        if not frame_path: return None
        img = cv2.imread(frame_path)
        if img is None: return None
        img = img[:, :, ::-1]  # BGR to RGB
        img = cv2.resize(img, (W, H))
        tensor = normalize_transform(img).to(device)
        return tensor
    except Exception as e:
        print(f"Warning: Failed to load frame {frame_path}: {e}")
        return None

def make_feather_mask(ph, pw, edge_soft=0.15, device="cpu"):
    """创建边缘羽化的蒙版"""
    yy = torch.linspace(-1, 1, ph, device=device).unsqueeze(1)
    xx = torch.linspace(-1, 1, pw, device=device).unsqueeze(0)
    dist = torch.max(torch.abs(xx), torch.abs(yy))
    mask = 1.0 - torch.clamp((dist - (1.0 - edge_soft)) / (edge_soft + 1e-8), 0, 1)
    return mask.clamp(0, 1)

# --- 策略 1: 基于补丁的伪异常 (Patch-based) ---
def apply_patch_replace(tgt_frame, source_video_frames, H, W, device,
                        num_patches=(3, 6),
                        patch_size_scale=(0.3, 0.8),
                        edge_soft=0.0):
    
    if not source_video_frames: return tgt_frame
    
    n_patches = random.randint(num_patches[0], num_patches[1])
    out_frame = tgt_frame.clone()

    for _ in range(n_patches):
        try:
            rand_frame_path = random.choice(source_video_frames)
            rand_frame = load_frame_as_tensor(rand_frame_path, H, W, device)
            if rand_frame is None: continue

            # 随机尺寸
            scale = random.uniform(patch_size_scale[0], patch_size_scale[1])
            ph = min(H, max(8, int(round(32 * scale * 2)))) # 基准32扩大
            pw = min(W, max(8, int(round(32 * scale * 2))))
            
            if H - ph <= 0 or W - pw <= 0: continue

            # 随机裁剪源
            sy = random.randint(0, H - ph)
            sx = random.randint(0, W - pw)
            patch = rand_frame[:, sy:sy + ph, sx:sx + pw]

            if random.random() < 0.7:
                # 亮度/对比度抖动
                jitter = (torch.rand(1, device=device) * 0.4 + 0.8) # 0.8 ~ 1.2
                patch = (patch * jitter).clamp(-1, 1)
            
            if random.random() < 0.3:
                # 加入微量噪声
                noise = torch.randn_like(patch) * 0.05
                patch = (patch + noise).clamp(-1, 1)

            # 随机目标位置
            ty = random.randint(0, H - ph)
            tx = random.randint(0, W - pw)

            # 羽化蒙版
            mask = make_feather_mask(ph, pw, edge_soft=edge_soft, device=device)
            alpha = random.uniform(0.8, 1.0)
            alpha_mask = (mask * alpha).unsqueeze(0)

            # 粘贴
            patch_crop = patch
            region = out_frame[:, ty:ty+ph, tx:tx+pw]
            if region.shape == patch_crop.shape:
                blended = patch_crop * alpha_mask + region * (1 - alpha_mask)
                out_frame[:, ty:ty+ph, tx:tx+pw] = blended

        except Exception as e:
            pass
            
    return out_frame

# --- 策略 2: 基于跳帧/重复帧的伪异常 (Temporal Shift) ---
def apply_temporal_shift(current_idx, video_frames, H, W, device, mode='skip'):
    try:
        total_frames = len(video_frames)
        if total_frames < 10: return None

        if mode == 'skip':
            # 跳帧：本来该第 t 帧，结果换成了 t + s 帧 (模拟快进/突变)
            # s 在 [5, 30] 之间随机，幅度要够大
            skip_step = random.randint(5, 30)
            target_idx = min(current_idx + skip_step, total_frames - 1)
        else: # mode == 'repeat'
            # 重复帧：本来该第 t 帧，结果还是 t-1 或 t-2 帧 (模拟停滞)
            repeat_step = random.randint(1, 5)
            target_idx = max(0, current_idx - repeat_step)
        
        if target_idx == current_idx: return None

        path = video_frames[target_idx]
        frame = load_frame_as_tensor(path, H, W, device)
        return frame
    except Exception:
        return None

# --- 策略 3: 基于融合的伪异常 (Blending) ---
def apply_blending(tgt_frame, source_video_frames, H, W, device):
    if not source_video_frames: return tgt_frame
    try:
        # 随机选一帧
        rand_frame_path = random.choice(source_video_frames)
        rand_frame = load_frame_as_tensor(rand_frame_path, H, W, device)
        if rand_frame is None: return tgt_frame

        # 随机混合比例 alpha
        alpha = random.uniform(0.2, 0.6)
        
        # I_p = (I_1 + I_2) / 2 或加权
        out_frame = tgt_frame * (1 - alpha) + rand_frame * alpha
        return out_frame
    except Exception:
        return tgt_frame


def prepare_batch_with_random_pseudo(x_clean, video_names, start_indices, dataset, apply_threshold=0.2):
    if not hasattr(dataset, 'video_map') or not dataset.video_map:
        return x_clean, torch.zeros(x_clean.shape[0], dtype=torch.bool, device=x_clean.device)

    B, T, C, H, W = x_clean.shape
    device = x_clean.device
    x_out = x_clean.clone() 
    
    p_vals = torch.rand(B, device=device)
    is_pseudo_mask = (p_vals < apply_threshold)

    for i in range(B):
        if not is_pseudo_mask[i]: 
            continue

        current_video_name = video_names[i]
        current_frames = dataset.video_map.get(current_video_name, [])
        start_idx_int = start_indices[i].item() if isinstance(start_indices[i], torch.Tensor) else start_indices[i]
        
        exclude_range = 30
        exclude = set(range(max(0, start_idx_int - exclude_range), 
                            min(len(current_frames), start_idx_int + T + exclude_range)))
        source_frames = [f for idx, f in enumerate(current_frames) if idx not in exclude]
        if not source_frames: 
            continue

        strategy_rand = random.random()

        try:
            if strategy_rand < 0.7:  # Patch-based
                base_frame = x_clean[i, T // 2]  # (C, H, W)
                patched_frame = apply_patch_replace(
                    base_frame,
                    source_video_frames=source_frames,
                    H=H, W=W, device=device,
                    num_patches=(3, 6),
                    patch_size_scale=(0.05, 0.45),
                    edge_soft=0.15
                )
                x_out[i] = patched_frame.unsqueeze(0).expand(T, C, H, W)

            elif strategy_rand < 0.9:  # Blending
                for t in range(T):
                    x_out[i, t] = apply_blending(
                        x_out[i, t],
                        source_video_frames=source_frames,
                        H=H, W=W, device=device
                    )

            else:  # Temporal Shift
                for t in range(T):
                    current_abs_idx = start_idx_int + t
                    shifted_frame = apply_temporal_shift(
                        current_idx=current_abs_idx,
                        video_frames=current_frames,
                        H=H, W=W, device=device,
                        mode=random.choice(['skip', 'repeat'])
                    )
                    if shifted_frame is not None:
                        x_out[i, t] = shifted_frame

        except Exception as e:
            print(f"HPAS error on sample {i}: {e}")

    return x_out, is_pseudo_mask
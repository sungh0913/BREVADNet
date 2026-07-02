# tools.py
import os
import argparse
import math
from copy import deepcopy
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import k_diffusion as K
import random
import numpy as np
import accelerate
import matplotlib.pyplot as plt
import glob
from torchmetrics.functional import auroc
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from autoencoder.data_load import SlidingWindowDataset

def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_video_names_list(video_names):
    if isinstance(video_names, torch.Tensor):
        return [v if isinstance(v, str) else v.item() for v in video_names]
    elif isinstance(video_names, str):
        return [video_names]
    elif isinstance(video_names, list):
        return video_names
    else:
        raise TypeError(f"Unexpected video_names type: {type(video_names)}")

def check_model_health(model, model_ema, vae):
    def check_params(param_dict, name):
        for n, p in param_dict:
            if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
                print(f"警告: {name} 中检测到 NaN/Inf: {n}")
                return False
        return True
    model_ok = check_params(model.named_parameters(), "Model")
    model_ema_ok = check_params(model_ema.named_parameters(), "Model EMA")
    vae_ok = check_params(vae.named_parameters(), "VAE")
    return model_ok and model_ema_ok and vae_ok

def save_checkpoint_with_rotation(ckpt, base_path, keep_last_n=3):
    existing_ckpts = glob.glob(f"{base_path}_epoch_*.pth")
    try:
        existing_ckpts.sort(key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
    except Exception:
        existing_ckpts.sort()
    
    if len(existing_ckpts) >= keep_last_n:
        num_to_delete = len(existing_ckpts) - (keep_last_n - 1)
        for old_ckpt in existing_ckpts[:num_to_delete]:
            Path(old_ckpt).unlink()

    epoch = ckpt.get('epoch', 0)
    new_path = f"{base_path}_epoch_{epoch}.pth"
    torch.save(ckpt, new_path)
    print(f"Saved checkpoint to {new_path}")
    return new_path

def save_training_plots(epoch_list, loss_dict, raw_metrics_dict=None, auc_log=None, save_dir='.', args=None):
    if not epoch_list: return

    fig, ax1 = plt.subplots(figsize=(12, 8))
    if 'total' in loss_dict: ax1.plot(epoch_list, loss_dict['total'], 'b-', label='Total Loss')
    if 'latent_normal_loss' in loss_dict: ax1.plot(epoch_list, loss_dict['latent_normal_loss'], 'g--', label='Latent Normal MSE')
    if 'latent_pseudo_loss' in loss_dict: ax1.plot(epoch_list, loss_dict['latent_pseudo_loss'], 'c-.', label='Latent Pseudo Loss')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))
    plt.close(fig)

    if auc_log:
        try:
            auc_epochs, auc_scores = zip(*auc_log)
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(auc_epochs, [s * 100 for s in auc_scores], 'm-o', label='AUC (%)')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('AUC (%)')
            ax1.legend()
            ax1.grid(True)
            plt.savefig(os.path.join(save_dir, 'evaluation_auc.png'))
            plt.close(fig)
        except Exception: pass

def denormalize_batch(tensor):
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)

def compute_gradient_difference(pred, target):
    blur_kernel_size = 5
    pred = F.avg_pool2d(pred, kernel_size=blur_kernel_size, stride=1, padding=blur_kernel_size//2)
    target = F.avg_pool2d(target, kernel_size=blur_kernel_size, stride=1, padding=blur_kernel_size//2)
    
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

    loss_dx = torch.abs(pred_dx - target_dx)
    loss_dy = torch.abs(pred_dy - target_dy)
    
    loss_dx = F.pad(loss_dx, (0, 1, 0, 0))
    loss_dy = F.pad(loss_dy, (0, 0, 0, 1))
    
    return (loss_dx + loss_dy).mean(dim=1)

@torch.no_grad()
def predict_latents_one_step(inference_model, unwrapped_vae, history_seq, future_frame_latents, args, device):
    B, T_hist, C, H, W = history_seq.shape
    C_z, H_z, W_z = future_frame_latents.shape[1:]
    
    sigma = getattr(args, 'start_sigma', 1.5) 
    sigmas = torch.tensor([sigma] * B, device=device)
    
    scale_factor = getattr(args, 'scale_factor', 0.18215)
    
    history_frames_flat = history_seq.view(B * T_hist, C, H, W)
    z_history_flat = unwrapped_vae.encode(history_frames_flat).sample() * scale_factor
    z_history_sequence = z_history_flat.view(B, T_hist, C_z, H_z, W_z)
    
    history_latents_input = z_history_sequence
    if args.no_temporal_cond: history_latents_input = None

    frame_t_minus_1 = history_seq[:, -1]
    frame_t_minus_2 = history_seq[:, -2]
    motion_map = frame_t_minus_1 - frame_t_minus_2
    condition_input = torch.cat([frame_t_minus_1, motion_map], dim=1)

    noise = torch.randn_like(future_frame_latents)
    noisy_latents = future_frame_latents + noise * sigma
    
    extra_args = {'model_cond': condition_input, 'history_latents': history_latents_input}
    
    predicted_x0 = inference_model(noisy_latents, sigmas, **extra_args)
    
    return predicted_x0

@torch.no_grad()
def evaluate(epoch, vae_ema, model_ema, accelerator, args):
    grad_weight = 1
    k_pro = 0.3
            
    if not accelerator.is_main_process:
        return -1.0

    print(f"\n--- Starting Evaluation (Metric: MSE + Gradient) ---")
    device = accelerator.device
    
    current_scale_factor = 0.18215
    if hasattr(args, 'config') and args.config:
        try:
            tmp_config = K.config.load_config(open(args.config))
            current_scale_factor = tmp_config['model'].get('scale_factor', 0.18215)
        except Exception as e:
            print(f"[Warning] Could not load scale_factor from config in evaluate: {e}")
    
    args.scale_factor = current_scale_factor
    scale_factor = args.scale_factor 
    
    print(f"[Eval] Using Dynamic Scale Factor: {scale_factor}")
    
    current_ds_name = args.dataset_name.lower() if args.dataset_name else 'ped2'
    
    output_dir_dataset = f'{args.output_dir}_{args.dataset_name}'
    epoch_output_dir = os.path.join(output_dir_dataset, f'eval_{epoch}')
    scores_dir = os.path.join(epoch_output_dir, 'anomaly_scores')
    plots_dir = os.path.join(epoch_output_dir, 'score_plots')
    samples_dir = os.path.join(epoch_output_dir, 'sample_visualizations')
    
    os.makedirs(scores_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    if args.visualize_samples:
        os.makedirs(samples_dir, exist_ok=True)

    unwrapped_vae = accelerator.unwrap_model(vae_ema)
    raw_inner_model = model_ema.inner_model if hasattr(model_ema, 'inner_model') else model_ema
    if hasattr(raw_inner_model, 'module'): raw_inner_model = raw_inner_model.module
    
    if hasattr(model_ema, 'sigma_data'):
        inference_model = model_ema.__class__(raw_inner_model, sigma_data=model_ema.sigma_data)
    else:
        inference_model = model_ema
    inference_model.eval()

    test_videos_base_dir = f"./data/{args.dataset_name}/testing/frames"
    if not os.path.exists(test_videos_base_dir):
        print(f"Error: Path {test_videos_base_dir} not found.")
        return -1.0
    video_folders = sorted([d.path for d in os.scandir(test_videos_base_dir) if d.is_dir()])
    
    label_path = f'./data/{args.dataset_name}/frame_labels_{args.dataset_name}.npy'
    all_gt_labels_flat = None
    label_map = {} 
    
    try:
        if os.path.exists(label_path):
            all_gt_labels_flat = np.load(label_path, allow_pickle=True).flatten()
            current_ptr = 0
            for vp in video_folders:
                vn = os.path.basename(vp)
                n_frames = len(glob.glob(os.path.join(vp, '*.jpg')))
                label_map[vn] = current_ptr
                current_ptr += n_frames
    except Exception as e:
        print(f"GT Load Error: {e}")

    valid_scores_for_auc = []
    valid_gt_for_auc = []
    final_scores_dict = {} 
    label_pointer = 0
    pad_len = args.clip_len - 1

    for video_path in tqdm(video_folders, desc="Evaluating"):
        video_name = os.path.basename(video_path)
        frame_paths = sorted(glob.glob(os.path.join(video_path, '*.jpg')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        n_frames = len(frame_paths)
        
        current_gt = None
        if all_gt_labels_flat is not None:
            s_idx = label_pointer
            e_idx = label_pointer + n_frames
            if e_idx <= len(all_gt_labels_flat):
                current_gt = all_gt_labels_flat[s_idx:e_idx]
            else:
                current_gt = all_gt_labels_flat[s_idx:]
            
            if current_gt is not None and len(current_gt) > pad_len:
                current_gt = current_gt[pad_len:]
            else:
                current_gt = np.array([])
        
        label_pointer += n_frames
        if n_frames < args.clip_len: continue

        dataset = SlidingWindowDataset(os.path.dirname(frame_paths[0]), args.clip_len, recursive=False, cache_to_ram=False)
        dl = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)
        
        video_mse_list = []
        for batch in dl:
            if batch is None: continue
            hist = batch['history_sequence'].to(device)
            future = batch['future_frame'].to(device)
            
            gt_latents = unwrapped_vae.encode(future).sample() * scale_factor
            pred_latents = predict_latents_one_step(inference_model, unwrapped_vae, hist, gt_latents, args, device)
            
            with torch.no_grad():
                pred_imgs = unwrapped_vae.decode(pred_latents / scale_factor)
                
                pred_imgs = torch.clamp((pred_imgs + 1.0) / 2.0, 0.0, 1.0)
                gt_imgs = torch.clamp((future + 1.0) / 2.0, 0.0, 1.0)
                
                mse_map = torch.mean(torch.pow(pred_imgs - gt_imgs, 2), dim=1) 
                
                grad_map = compute_gradient_difference(pred_imgs, gt_imgs) 
                
                anomaly_map = mse_map + grad_weight * grad_map
                
                
                B_batch, H_batch, W_batch = anomaly_map.shape
                flat_map = anomaly_map.view(B_batch, -1)
                k = max(1, int(H_batch * W_batch * k_pro))
                score = torch.topk(flat_map, k, dim=1).values.mean(dim=1)
                video_mse_list.extend(score.cpu().tolist())
            
        if not video_mse_list: continue
        
        raw_scores = np.array(video_mse_list, dtype=np.float32)
        raw_scores = gaussian_filter1d(raw_scores, sigma=4.0)

        v_max = np.percentile(raw_scores, 99.9)
        v_max = max(v_max, 1e-5) 

        v_min = raw_scores.min()
        norm_scores = (raw_scores - v_min) / (v_max - v_min)
            
        norm_scores = np.clip(norm_scores, 0.0, 1.0)
        
        final_scores_dict[video_name] = norm_scores

        np.savetxt(os.path.join(scores_dir, f"{video_name}_scores.txt"), norm_scores, fmt='%.6f')
        
        min_len = min(len(norm_scores), len(current_gt))
        if min_len > 0:
            valid_scores_for_auc.append(norm_scores[:min_len])
            valid_gt_for_auc.append(current_gt[:min_len])
            
            plt.figure(figsize=(12, 5)) 
            plt.plot(norm_scores[:min_len], 'b', label='Anomaly Score')
            plt.fill_between(range(min_len), 0, 1, where=current_gt[:min_len]==1, color='r', alpha=0.3, label='GT')
            plt.title(f'{video_name} (MSE+Grad)')
            plt.xlabel(f'Frame Index (Offset by {pad_len})')
            plt.ylim(-0.05, 1.05)
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(plots_dir, f"{video_name}_score.png"))
            plt.close()
            
    auc_score = -1.0
    if len(valid_gt_for_auc) > 0:
        flat_gt = np.concatenate(valid_gt_for_auc)
        flat_scores = np.concatenate(valid_scores_for_auc)
        try:
            auc_score = auroc(torch.from_numpy(flat_scores), torch.from_numpy(flat_gt).int(), task='binary').item()
            print(f"\n>>> Final AUC (MSE+Grad): {auc_score*100:.2f}% <<<")
            with open(os.path.join(output_dir_dataset, 'auc_log.txt'), 'a') as f:
                f.write(f"Epoch {epoch}: AUC = {auc_score*100:.2f} % (Metric: MSE+Grad)\n")
        except Exception as e:
            print(f"AUC Error: {e}")

    if args.visualize_samples:
        print("----- Visualizing Random Samples -----")
        vis_dataset = SlidingWindowDataset(test_videos_base_dir, args.clip_len, recursive=True, cache_to_ram=False)
        vis_dl = DataLoader(vis_dataset, batch_size=1, shuffle=True)
        count = 0
        
        for sample in vis_dl:
            if count >= args.num_samples_to_vis: break
            if sample is None: continue
            try:
                hist = sample['history_sequence'].to(device)
                fut = sample['future_frame'].to(device)
                vn = sample['video_name'][0]
                idx = sample['start_idx'][0].item() + args.clip_len - 1 
                
                with torch.no_grad():
                    gt_lat = unwrapped_vae.encode(fut).sample() * scale_factor
                    pred_lat = predict_latents_one_step(inference_model, unwrapped_vae, hist, gt_lat, args, device)
                    
                    pred_img = unwrapped_vae.decode(pred_lat / scale_factor)
                    
                    p_norm = torch.clamp((pred_img + 1.0) / 2.0, 0.0, 1.0)
                    g_norm = torch.clamp((fut + 1.0) / 2.0, 0.0, 1.0)
                    
                    mse_map = torch.mean(torch.pow(p_norm - g_norm, 2), dim=1)
                    grad_map = compute_gradient_difference(p_norm, g_norm)
                    amap = mse_map + grad_weight * grad_map
                    
                    B_v, H_v, W_v = amap.shape
                    k_v = max(1, int(H_v * W_v * k_pro))
                    raw_score_val = torch.topk(amap.view(B_v, -1), k_v, dim=1).values.mean().item()
                    
                    norm_val = -1.0
                    if vn in final_scores_dict:
                        scores_arr = final_scores_dict[vn]
                        arr_idx = idx - pad_len 
                        if 0 <= arr_idx < len(scores_arr):
                            norm_val = scores_arr[arr_idx]

                    pred_vis = denormalize_batch(pred_img).cpu().numpy()[0].transpose(1,2,0)
                    gt_vis = denormalize_batch(fut).cpu().numpy()[0].transpose(1,2,0)
                    
                    heatmap = amap[0].cpu().numpy()
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = plt.cm.jet(heatmap)[:, :, :3]

                label_str = "N/A"
                col = 'black'
                if vn in label_map and all_gt_labels_flat is not None:
                    abs_idx = label_map[vn] + idx
                    if abs_idx < len(all_gt_labels_flat):
                        is_ano = all_gt_labels_flat[abs_idx] == 1
                        label_str = "Anomaly" if is_ano else "Normal"
                        col = 'red' if is_ano else 'green'

                title_text = (
                    f"{vn} - Frame {idx}\n"
                    f"{label_str}\n"
                    f"Norm Score: {norm_val:.4f} | Raw Hybrid: {raw_score_val:.4f}"
                )

                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                fig.suptitle(title_text, color=col, fontsize=10, fontweight='bold')
                
                ax[0].imshow(gt_vis); ax[0].set_title("GT"); ax[0].axis('off')
                ax[1].imshow(pred_vis); ax[1].set_title("Prediction"); ax[1].axis('off')
                ax[2].imshow(heatmap); ax[2].set_title("MSE+Grad Map"); ax[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(samples_dir, f"{vn}_{idx}.png"))
                plt.close()
                count += 1
            except Exception as e: 
                print(f"Vis Error: {e}")
                continue

    return auc_score
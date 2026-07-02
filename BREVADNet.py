import torch
import torch.nn as nn
import torch.nn.functional as F
import k_diffusion as K

from autoencoder.Autoencoder import Autoencoder, Encoder, Decoder
from k_diffusion.models.fusion_unet import FusionUNet

class VideoAnomalyPredictionModel(nn.Module):
    def __init__(self, config, z_channels=3, emb_channels=3, scale_factor=0.18215):
        super().__init__()
        self.scale_factor = scale_factor

        enc = Encoder(
            channels=128, channel_multipliers=[1, 2, 4], n_resnet_blocks=2,
            in_channels=3, z_channels=z_channels
        )
        dec = Decoder(
            channels=128, channel_multipliers=[1, 2, 4], n_resnet_blocks=2,
            out_channels=3, z_channels=z_channels
        )
        self.vae = Autoencoder(enc, dec, emb_channels=emb_channels, z_channels=z_channels)

        inner_model = FusionUNet(config)
        self.diffusion_model = K.config.make_denoiser_wrapper(config)(inner_model)

    def encode_history(self, history_frames):
        B, T, C, H, W = history_frames.shape
        history_flat = history_frames.view(B * T, C, H, W)

        z_history_flat = self.vae.encode(history_flat).sample() * self.scale_factor

        _, C_z, H_z, W_z = z_history_flat.shape
        z_history_sequence = z_history_flat.view(B, T, C_z, H_z, W_z)
        return z_history_sequence

    def get_conditions(self, history_frames):
        frame_t_minus_1 = history_frames[:, -1]
        frame_t_minus_2 = history_frames[:, -2]
        motion_map = frame_t_minus_1 - frame_t_minus_2

        condition_input = torch.cat([frame_t_minus_1, motion_map], dim=1)
        return condition_input

    def forward(self, history_frames, noisy_future_latents, sigmas):
        with torch.no_grad():
            history_latents = self.encode_history(history_frames)

        condition_input = self.get_conditions(history_frames)

        predicted_output = self.diffusion_model(
            input=noisy_future_latents,
            sigma=sigmas,
            model_cond=condition_input,
            history_latents=history_latents
        )
        return predicted_output

    def predict_future_frame(self, history_frames, future_frame_gt, sigmas):
        B = history_frames.shape[0]
        device = history_frames.device

        history_latents = self.encode_history(history_frames)
        condition_input = self.get_conditions(history_frames)

        z_future_gt = self.vae.encode(future_frame_gt).sample() * self.scale_factor

        noise = torch.randn_like(z_future_gt)
        noisy_latents = z_future_gt + noise * sigmas.view(-1, 1, 1, 1)

        predicted_latents = self.diffusion_model(
            input=noisy_latents,
            sigma=sigmas,
            model_cond=condition_input,
            history_latents=history_latents
        )

        predicted_frame = self.vae.decode(predicted_latents / self.scale_factor)

        predicted_frame = torch.clamp((predicted_frame + 1.0) / 2.0, 0.0, 1.0)

        return predicted_frame

# 伪异常处理 (HPAS) 位置说明：
#         HPAS（历史伪异常生成）是在前置数据增强阶段完成的，即在输入网络之前、在像素级历史帧上直接
#        进行替换与修改，不是在编码器（Encoder）将历史帧转化为潜在特征（Latent Features）之后进行的。
#        因此，前向传播流中不包含任何 HPAS 的逻辑代码。
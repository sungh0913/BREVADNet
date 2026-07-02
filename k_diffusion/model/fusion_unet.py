import torch
import torch.nn as nn
import torch.nn.functional as F
from .openaimodel import UNetModel

from module.LSDC_Modified import LSDC
from module.ADCB import ADCB

class Adapter(nn.Module):
    def __init__(self, in_channels=32, base_channels=32):
        super().__init__()
        def make_layer(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                nn.GroupNorm(8, cout),
                nn.SiLU()
            )

        self.down0 = make_layer(in_channels, base_channels)
        self.down1 = make_layer(base_channels, base_channels)
        self.down2 = make_layer(base_channels, base_channels)
        self.down3 = make_layer(base_channels, base_channels)

    def forward(self, x):
        f_128 = self.down0(x)
        f_64 = self.down1(f_128)
        f_32 = self.down2(f_64)
        f_16 = self.down3(f_32)
        return [f_16, f_32, f_64, f_128, x]

LSDCAdapter = Adapter

class FusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_conf = config['model']
        
        self.lsdc = LSDC(
            in_channels=model_conf['z_channels'], 
            out_channels=32, 
            clip_len=model_conf['clip_len'] - 1
        )
        self.lsdc_adapter = LSDCAdapter(in_channels=32, base_channels=32)
        
        self.adcb = ADCB(in_channels=6, base_channels=32)
        self.adcb_adapter = Adapter(in_channels=32, base_channels=32)

        self.main_unet = UNetModel(
            image_size=model_conf['input_size'],
            in_channels=model_conf.get('in_channels', 4),
            out_channels=model_conf['out_channels'],
            model_channels=model_conf['ch'],
            num_res_blocks=model_conf['num_res_blocks'],
            attention_resolutions=model_conf['attn_resolutions'],
            dropout=model_conf['dropout'],
            channel_mult=model_conf['ch_mult'],
            conv_resample=True,
            dims=2,
            num_heads=model_conf.get('num_heads', 8),
            use_scale_shift_norm=True,
            resblock_updown=True,
            lsdc_channels=32,
            adcb_channels=32
        )

    def forward(self, x, t, model_cond=None, history_latents=None, **kwargs):
        # --- 1. 提取 LSDC 特征 ---
        lsdc_feats_list = None 
        if history_latents is not None:
            with torch.amp.autocast('cuda', enabled=False):
                hist_in = history_latents.float()
                if hist_in.ndim == 4:
                    B, C_total, H, W = hist_in.shape
                    T = self.lsdc.clip_len
                    C = C_total // T
                    hist_in = hist_in.view(B, T, C, H, W)
                lsdc_feature_raw = self.lsdc(hist_in)
                lsdc_feats_list = self.lsdc_adapter(lsdc_feature_raw)

        # --- 2. 提取 ADCB 多尺度特征 ---
        adcb_feats_list = None
        if model_cond is not None:
            with torch.amp.autocast('cuda', enabled=False):
                cond_in = model_cond.float()
                raw_adcb = self.adcb(cond_in) 
                adcb_feats_list = self.adcb_adapter(raw_adcb)
                
        # --- 3. 注入 U-Net ---
        return self.main_unet(
            x, t, 
            lsdc_feats=lsdc_feats_list, 
            adcb_feats=adcb_feats_list,
            **kwargs
        )
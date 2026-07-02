import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicBranch(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DynamicBranch, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        mid_channels = max(in_channels // reduction, 4) 
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        w = self.sigmoid(avg_out + max_out) 
        return w

class StaticBranch(nn.Module):
    def __init__(self, in_channels, dilations=[1, 3, 5],groups=4):
        super(StaticBranch, self).__init__()
        self.dilations = dilations
        self.num_branches = len(dilations)
        self.in_channels = in_channels 

        self.branches = nn.ModuleList()
        for d in self.dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.in_channels, 
                        out_channels=self.in_channels, 
                        kernel_size=3, 
                        padding=d,
                        dilation=d, 
                        groups=self.in_channels, 
                        bias=False
                    ),
                    nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=2, num_channels=self.in_channels),
                    nn.SiLU(inplace=True)
                )
            )
        
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // 4, 4), kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(max(in_channels // 4, 4), self.num_branches, kernel_size=1)
        )
        self.gate = nn.Softmax(dim=1) 
        
        self.groups = groups
        self.compress_conv = nn.Conv2d(self.in_channels, self.groups, 1, bias=False)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x)) 
        
        dyn_weights = self.weight_generator(x)
        dyn_weights = self.gate(dyn_weights)
        
        att_fused = torch.zeros_like(x) 
        for i in range(self.num_branches):
            att_fused += branch_outputs[i] * dyn_weights[:, i:i+1]
            
        att = self.compress_conv(att_fused) # [B, G, H, W]
        att = self.final_sigmoid(att)

        B, C, H, W = x.shape
        att = att.unsqueeze(2).expand(B, self.groups, C // self.groups, H, W).reshape(B, C, H, W)
        return att

class LSDC(nn.Module):
    def __init__(self, in_channels, out_channels, clip_len=4):
        super().__init__()
        self.clip_len = clip_len
        self.in_channels = in_channels
        self.out_channels = out_channels

        fused_channels = in_channels * clip_len

        self.dy_branch = DynamicBranch(fused_channels, reduction=16) 

        self.st_branch = StaticBranch(fused_channels, dilations=[1, 3, 5])
        self.output_conv = nn.Conv2d(in_channels * clip_len, 32, kernel_size=1)

    def forward(self, z_history_sequence):
        B, T, C, H, W = z_history_sequence.shape

        if T != self.clip_len:
            z_history_sequence = F.interpolate(
                z_history_sequence.permute(0, 2, 1, 3, 4),
                size=(self.clip_len, H, W), 
                mode='trilinear', 
                align_corners=False
            ).permute(0, 2, 1, 3, 4)
            T = self.clip_len
        
        x = z_history_sequence.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, C * T, H, W)
        
        w_dy = self.dy_branch(x)
        w_st = self.st_branch(x)
        
        x_gated = x + x * w_dy * w_st

        spatial_context = self.output_conv(x_gated)

        return spatial_context
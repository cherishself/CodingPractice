import torch

class MultiLevelLinearHeads(torch.nn.Module):
    def __init__(self, in_dims, out_dim=256):
        super().__init__()
        # 使用多个线性层对齐不同层次（论文用 6/12/18/24 层）
        self.heads = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d, out_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(out_dim, out_dim)
            ) for d in in_dims
        ])

    def forward(self, feats_list):
        # feats_list: list of (B, D_l) or (B, N, D_l)
        aligned = []
        for head, feats in zip(self.heads, feats_list):
            if feats.dim() == 3:  # (B, N, D)
                # 对序列维度进行平均池化
                feats = feats.mean(dim=1)  # (B, D)
            aligned.append(head(feats))
        
        # 融合平均
        fused = torch.stack(aligned, dim=0).mean(dim=0)  # (B, out_dim)
        return fused

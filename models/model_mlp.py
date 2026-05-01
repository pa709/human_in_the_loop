
import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, T_max, in_channels=6, out_channels=6, hidden=64):
        super().__init__()
        self.T_max = T_max
        self.in_channels = in_channels
        self.out_channels = out_channels
        in_dim = T_max * in_channels + 1    # +1 for sequence length
        out_dim = T_max * out_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, X, lengths):
        """
        X: (B, T_max, in_channels) — zero-padded human trajectory
        lengths: (B,) — real sequence lengths
        Returns: (B, T_max, out_channels) — predicted robot trajectory
        """
        B, T, _ = X.shape
        if T < self.T_max:
            pad = torch.zeros(B, self.T_max - T, self.in_channels, device=X.device, dtype=X.dtype)
            X = torch.cat([X, pad], dim=1)
        flat = X.reshape(B, -1)                                  # (B, T_max*6)
        # normalize length to [0,1] range so it's on similar scale to pose data
        norm_len = (lengths.float() / self.T_max).unsqueeze(1)   # (B, 1)
        inp = torch.cat([flat, norm_len], dim=1)                 # (B, T_max*6 + 1)
        out = self.net(inp)                                      # (B, T_max*6)
        return out.reshape(B, self.T_max, self.out_channels)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

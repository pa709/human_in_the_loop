
import torch
import torch.nn as nn

class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim=6, out_dim=6, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.out_dim = out_dim
        self.encoder = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(out_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def encode(self, X):
        _, (h, c) = self.encoder(X)
        return h, c

    def forward(self, X, Y=None, teacher_force=True):
        """
        X: (B, T, in_dim) encoder input
        Y: (B, T, out_dim) ground-truth decoder output (for teacher forcing/loss).
           If None or teacher_force=False, decode autoregressively.
        Returns: (B, T, out_dim) predictions.
        """
        B, T, _ = X.shape
        h, c = self.encode(X)

        if teacher_force and Y is not None:
            # decoder input: zero BOS prepended, last GT step dropped
            bos = torch.zeros(B, 1, self.out_dim, device=X.device, dtype=X.dtype)
            dec_in = torch.cat([bos, Y[:, :-1, :]], dim=1)   # (B, T, out_dim)
            dec_out, _ = self.decoder(dec_in, (h, c))         # (B, T, hidden)
            return self.head(dec_out)
        else:
            # autoregressive
            outs = []
            prev = torch.zeros(B, 1, self.out_dim, device=X.device, dtype=X.dtype)
            hh, cc = h, c
            for _ in range(T):
                step, (hh, cc) = self.decoder(prev, (hh, cc))
                pred = self.head(step)            # (B, 1, out_dim)
                outs.append(pred)
                prev = pred
            return torch.cat(outs, dim=1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

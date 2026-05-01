
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

    def forward(self, X, Y=None, teacher_force=True, sample_prob=0.0):
        """
        X: (B, T, in_dim) encoder input
        Y: (B, T, out_dim) ground-truth decoder output (for teacher forcing/loss).
           If None or teacher_force=False, decode autoregressively.
        sample_prob: float in [0,1]. Probability of using own prediction instead of
                     ground truth at each decoder step (scheduled sampling).
                     0.0 = pure teacher forcing, 1.0 = pure autoregressive.
        Returns: (B, T, out_dim) predictions.
        """
        B, T, _ = X.shape
        h, c = self.encode(X)

        if teacher_force and Y is not None and sample_prob == 0.0:
            # pure teacher forcing — fast batched path
            bos = torch.zeros(B, 1, self.out_dim, device=X.device, dtype=X.dtype)
            dec_in = torch.cat([bos, Y[:, :-1, :]], dim=1)   # (B, T, out_dim)
            dec_out, _ = self.decoder(dec_in, (h, c))         # (B, T, hidden)
            return self.head(dec_out)
        else:
            # step-by-step: autoregressive or scheduled sampling
            outs = []
            prev = torch.zeros(B, 1, self.out_dim, device=X.device, dtype=X.dtype)
            hh, cc = h, c
            for t in range(T):
                step, (hh, cc) = self.decoder(prev, (hh, cc))
                pred = self.head(step)            # (B, 1, out_dim)
                outs.append(pred)
                # decide next input: own prediction or ground truth
                if Y is not None and sample_prob < 1.0:
                    if t < T - 1:
                        use_pred = (torch.rand(1).item() < sample_prob)
                        prev = pred.detach() if use_pred else Y[:, t:t+1, :]
                    else:
                        prev = pred
                else:
                    prev = pred
            return torch.cat(outs, dim=1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import sys, os, json, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BlackoutPairs, collate_pad
from model_lstm import LSTMSeq2Seq, count_params

# --- training config ---
HIDDEN        = 32
EPOCHS        = 2000
BATCH_SIZE    = 8
LR            = 1e-3
WEIGHT_DECAY  = 0.0
ENDPOINT_FRAC = 0.10     # last 10% of timesteps...
ENDPOINT_W    = 3.0      # ...weighted this much
LOG_EVERY     = 50
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# --- scheduled sampling ---
# For the first SS_WARMUP epochs: pure teacher forcing (sample_prob=0).
# Then linearly ramp sample_prob from 0 to SS_MAX over the remaining epochs.
SS_WARMUP     = 500      # pure teacher-forcing warmup epochs
SS_MAX        = 0.5      # max probability of using own prediction per step


def make_step_weights(lengths, T_max, frac, w):
    """Per-step weight tensor (B, T, 1). 1.0 everywhere except final `frac` of
       each sequence's valid region, which is `w`. Padded steps are 0."""
    B = len(lengths)
    W = torch.zeros(B, T_max, 1)
    for i, L in enumerate(lengths.tolist()):
        W[i, :L, 0] = 1.0
        anchor_start = max(0, int(L * (1.0 - frac)))
        W[i, anchor_start:L, 0] = w
    return W


def weighted_mse(pred, target, weights):
    """pred, target: (B, T, D). weights: (B, T, 1). Returns scalar."""
    se = (pred - target) ** 2          # (B,T,D)
    se = se.mean(dim=-1, keepdim=True) # (B,T,1) avg over D
    num = (se * weights).sum()
    den = weights.sum().clamp_min(1e-8)
    return num / den


def channel_l2(pred, target, lengths, channels):
    """Mean L2 error over selected channels, averaged over valid timesteps."""
    # pred,target: (B,T,D); channels: list of dim indices
    p = pred[..., channels]
    t = target[..., channels]
    err = torch.linalg.vector_norm(p - t, dim=-1)   # (B,T)
    total = 0.0; count = 0
    for i, L in enumerate(lengths.tolist()):
        total += err[i, :L].sum().item()
        count += L
    return total / max(count, 1)


def endpoint_l2(pred, target, lengths, channels):
    """L2 error on the final valid timestep only, averaged across batch."""
    vals = []
    for i, L in enumerate(lengths.tolist()):
        p = pred[i, L - 1, channels]
        t = target[i, L - 1, channels]
        vals.append(torch.linalg.vector_norm(p - t).item())
    return float(np.mean(vals))


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 train.py PREPROC.hdf5 ROBOT.hdf5 OUT_DIR")
        sys.exit(1)
    h_path, r_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    ds = BlackoutPairs(h_path, r_path)
    print(f"loaded {len(ds)} paired demos")
    print(f"first 3 demos: {[ds.samples[i][0] for i in range(min(3, len(ds)))]}")
    Ts = [len(s[1]) for s in ds.samples]
    print(f"sequence lengths: min={min(Ts)} max={max(Ts)} mean={np.mean(Ts):.1f}")

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)

    model = LSTMSeq2Seq(in_dim=6, out_dim=6, hidden=HIDDEN).to(DEVICE)
    n_params = count_params(model)
    print(f"model: LSTMSeq2Seq hidden={HIDDEN}, params={n_params}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    log = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # scheduled sampling: ramp sample_prob from 0 to SS_MAX after warmup
        if epoch <= SS_WARMUP:
            sample_prob = 0.0
        else:
            sample_prob = SS_MAX * (epoch - SS_WARMUP) / (EPOCHS - SS_WARMUP)

        ep_loss = 0.0; ep_pos = 0.0; ep_rot = 0.0; ep_endpos = 0.0; ep_endrot = 0.0
        n_batches = 0
        for names, X, Y, lengths in loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            W = make_step_weights(lengths, X.size(1), ENDPOINT_FRAC, ENDPOINT_W).to(DEVICE)

            pred = model(X, Y, teacher_force=True, sample_prob=sample_prob)
            loss = weighted_mse(pred, Y, W)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += loss.item()
            ep_pos += channel_l2(pred.detach(), Y, lengths, [0,1,2])
            ep_rot += channel_l2(pred.detach(), Y, lengths, [3,4,5])
            ep_endpos += endpoint_l2(pred.detach(), Y, lengths, [0,1,2])
            ep_endrot += endpoint_l2(pred.detach(), Y, lengths, [3,4,5])
            n_batches += 1

        ep_loss /= n_batches
        ep_pos /= n_batches; ep_rot /= n_batches
        ep_endpos /= n_batches; ep_endrot /= n_batches
        log.append({"epoch": epoch, "loss": ep_loss,
                    "pos_l2": ep_pos, "rot_l2": ep_rot,
                    "endpoint_pos_l2": ep_endpos, "endpoint_rot_l2": ep_endrot,
                    "sample_prob": round(sample_prob, 4)})

        if epoch == 1 or epoch % LOG_EVERY == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:5d}  loss={ep_loss:.5f}  "
                  f"pos_l2={ep_pos:.4f}  rot_l2={ep_rot:.4f}  "
                  f"end_pos={ep_endpos:.4f}  end_rot={ep_endrot:.4f}  "
                  f"ss={sample_prob:.2f}  ({time.time()-t0:.1f}s)")

    torch.save({"model_state": model.state_dict(),
                "config": {"hidden": HIDDEN, "in_dim": 6, "out_dim": 6}},
               os.path.join(out_dir, "seq2seq_model.pt"))
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved model + log to {out_dir}")


if __name__ == "__main__":
    main()


import sys
import numpy as np
import torch

from dataset import BlackoutPairs, collate_pad
from model_mlp import MLPBaseline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def per_step_l2(pred, target, channels):
    p = pred[..., channels]; t = target[..., channels]
    return torch.linalg.vector_norm(p - t, dim=-1)


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 evaluate_mlp.py PREPROC.hdf5 ROBOT.hdf5 model.pt")
        sys.exit(1)
    h_path, r_path, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]

    ds = BlackoutPairs(h_path, r_path)
    Ts = [len(s[1]) for s in ds.samples]
    T_max = max(Ts)

    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    cfg = state["config"]
    model = MLPBaseline(T_max=cfg["T_max"], in_channels=cfg["in_channels"],
                        out_channels=cfg["out_channels"], hidden=cfg["hidden"]).to(DEVICE)
    model.load_state_dict(state["model_state"])
    model.eval()

    print(f"loaded MLP model: hidden={cfg['hidden']}, T_max={cfg['T_max']}, demos={len(ds)}")
    print()
    print(f"{'demo':<10}{'T':<5}|{'pos_l2':<10}{'rot_l2':<10}{'end_pos':<10}{'end_rot':<10}")
    print("-" * 60)

    rows = []
    with torch.no_grad():
        for name, X, Y in ds:
            T = len(X)
            # pad to T_max
            X_pad = torch.zeros(1, T_max, 6, device=DEVICE)
            X_pad[0, :T] = X
            Y_pad = torch.zeros(1, T_max, 6, device=DEVICE)
            Y_pad[0, :T] = Y
            lengths = torch.tensor([T], dtype=torch.long)

            pred = model(X_pad, lengths)

            pos = per_step_l2(pred[0, :T], Y_pad[0, :T], [0,1,2]).mean().item()
            rot = per_step_l2(pred[0, :T], Y_pad[0, :T], [3,4,5]).mean().item()
            ep  = per_step_l2(pred[0, T-1:T], Y_pad[0, T-1:T], [0,1,2]).item()
            er  = per_step_l2(pred[0, T-1:T], Y_pad[0, T-1:T], [3,4,5]).item()

            rows.append((pos, rot, ep, er))
            print(f"{name:<10}{T:<5}|{pos:<10.4f}{rot:<10.4f}{ep:<10.4f}{er:<10.4f}")

    rows = np.array(rows)
    print("-" * 60)
    print(f"{'MEAN':<10}{'':<5}|"
          f"{rows[:,0].mean():<10.4f}{rows[:,1].mean():<10.4f}"
          f"{rows[:,2].mean():<10.4f}{rows[:,3].mean():<10.4f}")


if __name__ == "__main__":
    main()

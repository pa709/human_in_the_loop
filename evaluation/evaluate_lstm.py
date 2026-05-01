import sys
import numpy as np
import torch

from dataset import BlackoutPairs
from model_lstm import LSTMSeq2Seq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def per_step_l2(pred, target, channels):
    p = pred[..., channels]; t = target[..., channels]
    return torch.linalg.vector_norm(p - t, dim=-1)   # (T,)


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 evaluate.py PREPROC.hdf5 ROBOT.hdf5 model.pt")
        sys.exit(1)
    h_path, r_path, ckpt = sys.argv[1], sys.argv[2], sys.argv[3]

    ds = BlackoutPairs(h_path, r_path)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    cfg = state["config"]
    model = LSTMSeq2Seq(in_dim=cfg["in_dim"], out_dim=cfg["out_dim"], hidden=cfg["hidden"]).to(DEVICE)
    model.load_state_dict(state["model_state"])
    model.eval()

    print(f"loaded model: hidden={cfg['hidden']}, demos={len(ds)}")
    print()
    print(f"{'demo':<10}{'T':<5}|{'TF pos':<10}{'TF rot':<10}{'TF end_p':<10}{'TF end_r':<10}|"
          f"{'AR pos':<10}{'AR rot':<10}{'AR end_p':<10}{'AR end_r':<10}")
    print("-" * 110)

    rows_tf = []; rows_ar = []
    with torch.no_grad():
        for name, X, Y in ds:
            X = X.unsqueeze(0).to(DEVICE)
            Y = Y.unsqueeze(0).to(DEVICE)

            pred_tf = model(X, Y, teacher_force=True)
            pred_ar = model(X, None, teacher_force=False)

            T = X.size(1)
            tf_pos = per_step_l2(pred_tf[0], Y[0], [0,1,2]).mean().item()
            tf_rot = per_step_l2(pred_tf[0], Y[0], [3,4,5]).mean().item()
            tf_ep  = per_step_l2(pred_tf[0, T-1:T], Y[0, T-1:T], [0,1,2]).item()
            tf_er  = per_step_l2(pred_tf[0, T-1:T], Y[0, T-1:T], [3,4,5]).item()

            ar_pos = per_step_l2(pred_ar[0], Y[0], [0,1,2]).mean().item()
            ar_rot = per_step_l2(pred_ar[0], Y[0], [3,4,5]).mean().item()
            ar_ep  = per_step_l2(pred_ar[0, T-1:T], Y[0, T-1:T], [0,1,2]).item()
            ar_er  = per_step_l2(pred_ar[0, T-1:T], Y[0, T-1:T], [3,4,5]).item()

            rows_tf.append((tf_pos, tf_rot, tf_ep, tf_er))
            rows_ar.append((ar_pos, ar_rot, ar_ep, ar_er))
            print(f"{name:<10}{T:<5}|"
                  f"{tf_pos:<10.4f}{tf_rot:<10.4f}{tf_ep:<10.4f}{tf_er:<10.4f}|"
                  f"{ar_pos:<10.4f}{ar_rot:<10.4f}{ar_ep:<10.4f}{ar_er:<10.4f}")

    rows_tf = np.array(rows_tf); rows_ar = np.array(rows_ar)
    print("-" * 110)
    print(f"{'MEAN':<10}{'':<5}|"
          f"{rows_tf[:,0].mean():<10.4f}{rows_tf[:,1].mean():<10.4f}"
          f"{rows_tf[:,2].mean():<10.4f}{rows_tf[:,3].mean():<10.4f}|"
          f"{rows_ar[:,0].mean():<10.4f}{rows_ar[:,1].mean():<10.4f}"
          f"{rows_ar[:,2].mean():<10.4f}{rows_ar[:,3].mean():<10.4f}")


if __name__ == "__main__":
    main()

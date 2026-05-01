import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

DROP_DEMOS = {"demo_40"}

class BlackoutPairs(Dataset):
    def __init__(self, human_path, robot_path):
        self.samples = []   # list of (name, X, Y) numpy arrays
        with h5py.File(human_path, "r") as fh, h5py.File(robot_path, "r") as fr:
            human_demos = set(fh["data"].keys())
            robot_demos = set(fr["data"].keys())
            common = sorted(human_demos & robot_demos, key=lambda x: int(x.split("_")[1]))
            for name in common:
                if name in DROP_DEMOS:
                    continue
                hg = fh["data"][name]
                rg = fr["data"][name]
                tv = hg["human_tvecs_scaled"][()]   # (T,3)
                rv = hg["human_rvecs_scaled"][()]   # (T,3)
                actions = rg["remaining_actions"][()]   # (T,7)
                T_h = len(tv); T_r = len(actions)
                if T_h != T_r:
                    raise ValueError(f"{name}: human T={T_h} vs robot T={T_r}")
                # input: delta from start of the scaled human trajectory
                tv_d = tv - tv[0]
                rv_d = rv - rv[0]
                X = np.concatenate([tv_d, rv_d], axis=1).astype(np.float32)   # (T,6)
                Y = actions[:, :6].astype(np.float32)                          # (T,6)
                self.samples.append((name, X, Y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        name, X, Y = self.samples[i]
        return name, torch.from_numpy(X), torch.from_numpy(Y)


def collate_pad(batch):
    """Pad variable-length sequences to the max T in the batch.
       Returns: names(list), X(B,Tmax,6), Y(B,Tmax,6), lengths(B,)."""
    names = [b[0] for b in batch]
    Xs = [b[1] for b in batch]
    Ys = [b[2] for b in batch]
    lengths = torch.tensor([len(x) for x in Xs], dtype=torch.long)
    Tmax = lengths.max().item()
    B = len(batch)
    Xpad = torch.zeros(B, Tmax, 6)
    Ypad = torch.zeros(B, Tmax, 6)
    for i, (x, y) in enumerate(zip(Xs, Ys)):
        Xpad[i, :len(x)] = x
        Ypad[i, :len(y)] = y
    return names, Xpad, Ypad, lengths

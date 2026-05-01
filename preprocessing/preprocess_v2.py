""

import sys
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


# --- tunables ---
BURST_DT_S        = 0.005
GAP_DT_S          = 0.100
LARGE_GAP_DT_S    = 0.500
EDGE_FRAC         = 0.10
NOMINAL_DT_S      = 1.0/30.0
STATIONARY_WIN    = 10
STATIONARY_STD_M  = 0.010
END_TAIL_FRAC     = 0.15
END_SPIKE_K       = 5.0
MIN_FINAL_LEN     = 15


def drop_startup_burst(ts, tv, rv):
    if len(ts) < 2:
        return ts, tv, rv, 0
    dt = np.diff(ts)
    drop = 0
    for d in dt:
        if d < BURST_DT_S:
            drop += 1
        else:
            break
    if drop == 0:
        return ts, tv, rv, 0
    return ts[drop:], tv[drop:], rv[drop:], drop


def snip_edge_gaps(ts, tv, rv):
    n = len(ts)
    if n < 3:
        return ts, tv, rv, 0, 0
    dt = np.diff(ts)
    edge_n = max(1, int(EDGE_FRAC * n))

    leading = 0
    leading_gaps = np.where(dt[:edge_n] > GAP_DT_S)[0]
    if len(leading_gaps) > 0:
        leading = leading_gaps[-1] + 1

    trailing = 0
    trailing_dt_indices = np.arange(max(0, n - 1 - edge_n), n - 1)
    trailing_gaps_in_window = np.where(dt[trailing_dt_indices] > GAP_DT_S)[0]
    if len(trailing_gaps_in_window) > 0:
        first_local = trailing_gaps_in_window[0]
        cut_at = trailing_dt_indices[first_local] + 1
        trailing = n - cut_at

    if leading == 0 and trailing == 0:
        return ts, tv, rv, 0, 0
    end = n - trailing
    return ts[leading:end], tv[leading:end], rv[leading:end], leading, trailing


def check_and_interpolate_mid_gaps(ts, tv, rv):
    if len(ts) < 2:
        return ts, tv, rv, 0
    dt = np.diff(ts)
    gap_idx = np.where(dt > GAP_DT_S)[0]
    if len(gap_idx) == 0:
        return ts, tv, rv, 0

    big = np.where(dt[gap_idx] > LARGE_GAP_DT_S)[0]
    if len(big) > 0:
        worst = dt[gap_idx[big[0]]]
        return None, None, None, f"mid-gap {worst*1000:.0f}ms > {int(LARGE_GAP_DT_S*1000)}ms"

    new_ts = [ts[0:1].copy()]
    new_tv = [tv[0:1].copy()]
    new_rv = [rv[0:1].copy()]
    n_inserted = 0
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        gap = t1 - t0
        if gap <= GAP_DT_S:
            new_ts.append(ts[i + 1:i + 2])
            new_tv.append(tv[i + 1:i + 2])
            new_rv.append(rv[i + 1:i + 2])
            continue
        n_insert = int(np.round(gap / NOMINAL_DT_S)) - 1
        if n_insert < 1:
            n_insert = 1
        alphas = np.linspace(0.0, 1.0, n_insert + 2)[1:-1]
        tv_interp = tv[i] + alphas[:, None] * (tv[i + 1] - tv[i])
        slerp = Slerp([0.0, 1.0], R.from_rotvec(np.stack([rv[i], rv[i + 1]])))
        rv_interp = slerp(alphas).as_rotvec()
        ts_interp = t0 + alphas * gap
        new_ts.append(ts_interp)
        new_tv.append(tv_interp)
        new_rv.append(rv_interp)
        n_inserted += n_insert
        new_ts.append(ts[i + 1:i + 2])
        new_tv.append(tv[i + 1:i + 2])
        new_rv.append(rv[i + 1:i + 2])

    return (np.concatenate(new_ts),
            np.concatenate(new_tv, axis=0),
            np.concatenate(new_rv, axis=0),
            n_inserted)


def find_motion_start(tv):
    n = len(tv)
    if n < STATIONARY_WIN + 2:
        return 0
    for i in range(n - STATIONARY_WIN):
        win = tv[i:i + STATIONARY_WIN]
        if np.max(np.std(win, axis=0)) > STATIONARY_STD_M:
            return i
    return 0


def trim_end_spike(tv):
    n = len(tv)
    if n < 10:
        return n
    var_axis = int(np.argmax(np.std(tv, axis=0)))
    series = tv[:, var_axis].copy()
    indices = np.arange(n)
    cutoff = int((1.0 - END_TAIL_FRAC) * n)
    if cutoff < 2:
        return n
    body_deltas = np.abs(np.diff(series[:cutoff]))
    if len(body_deltas) == 0:
        return n
    mu = float(np.mean(body_deltas))
    sigma = float(np.std(body_deltas))
    threshold = mu + END_SPIKE_K * sigma

    keep = np.ones(n, dtype=bool)
    while True:
        kept_idx = indices[keep]
        kept_series = series[keep]
        if len(kept_series) < 2:
            break
        deltas = np.abs(np.diff(kept_series))
        offenders = []
        for j, d in enumerate(deltas):
            later_orig_idx = kept_idx[j + 1]
            if later_orig_idx >= cutoff and d > threshold:
                offenders.append(later_orig_idx)
        if not offenders:
            break
        for oi in offenders:
            keep[oi] = False

    kept_idx = indices[keep]
    return int(kept_idx[-1] + 1)


def temporal_rescale(tv, rv, target_n):
    n = len(tv)
    if n == target_n:
        return tv.copy(), rv.copy()
    src_t = np.linspace(0.0, 1.0, n)
    tgt_t = np.linspace(0.0, 1.0, target_n)
    tv_out = np.stack([np.interp(tgt_t, src_t, tv[:, i]) for i in range(3)], axis=1)
    if n == 1:
        rv_out = np.tile(rv[0], (target_n, 1))
    else:
        slerp = Slerp(src_t, R.from_rotvec(rv))
        rv_out = slerp(tgt_t).as_rotvec()
    return tv_out, rv_out


def process_demo(ts, tv, rv, target_n):
    info = {"raw_n": len(ts)}

    ts, tv, rv, n_burst = drop_startup_burst(ts, tv, rv)
    info["dropped_burst"] = n_burst
    if len(ts) < MIN_FINAL_LEN:
        return {**info, "reject": "too short after burst"}

    ts, tv, rv, lead, trail = snip_edge_gaps(ts, tv, rv)
    info["edge_lead"] = lead
    info["edge_trail"] = trail
    if len(ts) < MIN_FINAL_LEN:
        return {**info, "reject": "too short after edge snip"}

    result = check_and_interpolate_mid_gaps(ts, tv, rv)
    if result[0] is None:
        return {**info, "reject": result[3]}
    ts, tv, rv, n_inserted = result
    info["interpolated"] = n_inserted

    start = find_motion_start(tv)
    info["motion_start"] = start
    end = trim_end_spike(tv)
    info["motion_end"] = end

    if end - start < MIN_FINAL_LEN:
        return {**info, "reject": f"snippet too short ({end - start})"}

    snippet_tv = tv[start:end]
    snippet_rv = rv[start:end]
    info["snippet_n"] = len(snippet_tv)

    tv_scaled, rv_scaled = temporal_rescale(snippet_tv, snippet_rv, target_n)
    info["tv_scaled"] = tv_scaled
    info["rv_scaled"] = rv_scaled
    info["target_n"] = target_n
    return info


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 preprocess_v2.py INPUT.hdf5 OUTPUT.hdf5")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    with h5py.File(in_path, "r") as fin, h5py.File(out_path, "w") as fout:
        d_in = fin["data"]
        d_out = fout.create_group("data")
        for k, v in d_in.attrs.items():
            d_out.attrs[k] = v

        demos = sorted(d_in.keys(), key=lambda x: int(x.split("_")[1]))

        header = (f"{'demo':<9}{'raw':<5}{'burst':<6}{'lead':<5}{'trail':<6}"
                  f"{'interp':<7}{'start':<6}{'end':<6}{'snip':<5}{'tgt':<5}verdict")
        print(header); print("-" * len(header))

        n_ok = n_skip = 0
        ok_demo_names = []
        for d in demos:
            g = d_in[d]
            ts = g["human_timestamps"][()]
            tv = g["human_tvecs"][()]
            rv = g["human_rvecs"][()]
            target = int(g.attrs["remaining_len"])

            info = process_demo(ts, tv, rv, target)

            row = (
                f"{d:<9}{info['raw_n']:<5}"
                f"{info.get('dropped_burst','-'):<6}"
                f"{info.get('edge_lead','-'):<5}"
                f"{info.get('edge_trail','-'):<6}"
                f"{info.get('interpolated','-'):<7}"
                f"{info.get('motion_start','-'):<6}"
                f"{info.get('motion_end','-'):<6}"
                f"{info.get('snippet_n','-'):<5}"
                f"{target:<5}"
            )

            if "reject" in info:
                print(row + f"REJECT: {info['reject']}")
                n_skip += 1
                continue

            print(row + "OK")
            g_out = d_out.create_group(d)
            for k, v in g.attrs.items():
                g_out.attrs[k] = v
            g_out.attrs["dropped_burst"] = info["dropped_burst"]
            g_out.attrs["edge_lead"] = info["edge_lead"]
            g_out.attrs["edge_trail"] = info["edge_trail"]
            g_out.attrs["interpolated"] = info["interpolated"]
            g_out.attrs["motion_start"] = info["motion_start"]
            g_out.attrs["motion_end"] = info["motion_end"]
            g_out.attrs["snippet_n"] = info["snippet_n"]
            g_out.create_dataset("human_tvecs_raw", data=tv)
            g_out.create_dataset("human_rvecs_raw", data=rv)
            g_out.create_dataset("human_timestamps_raw", data=ts)
            g_out.create_dataset("human_tvecs_scaled", data=info["tv_scaled"])
            g_out.create_dataset("human_rvecs_scaled", data=info["rv_scaled"])
            n_ok += 1
            ok_demo_names.append(d)

        print("-" * len(header))
        print(f"Wrote {n_ok}/{len(demos)} demos. Skipped {n_skip}. Output: {out_path}")
        print(f"Kept: {ok_demo_names}")


if __name__ == "__main__":
    main()

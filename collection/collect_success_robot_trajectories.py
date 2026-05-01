import argparse
import json
import h5py
import numpy as np
from copy import deepcopy
import os

import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy


def rollout_with_obs(policy, env, horizon, camera_name="agentview"):
    """
    Run one rollout, collecting actions, EEF poses, and RGB frames at every step.

    Returns:
        success (bool): whether the task was completed
        actions (np.ndarray): shape (T, 7) — OSC_POSE actions at each step
        eef_poses (np.ndarray): shape (T, 7) — EEF position (3) + quaternion (4) at each step
        rgb_frames (list of np.ndarray): RGB frame (H, W, 3) at each step, uint8
    """
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    actions = []
    eef_poses = []
    rgb_frames = []
    success = False

    try:
        for step_i in range(horizon):
            # get action from policy
            act = policy(ob=obs)

            # render RGB frame from agentview camera before stepping
            rgb = env.render(mode="rgb_array", height=256, width=256, camera_name=camera_name)
            rgb_frames.append(rgb.astype(np.uint8))

            # extract EEF pose from current observation
            # robosuite exposes robot0_eef_pos (3,) and robot0_eef_quat (4,)
            eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
            eef_quat = obs.get("robot0_eef_quat", np.zeros(4))
            eef_pose = np.concatenate([eef_pos, eef_quat])  # (7,)
            eef_poses.append(eef_pose)

            # record action
            actions.append(act.copy())

            # step environment
            next_obs, r, done, _ = env.step(act)
            success = env.is_success()["task"]

            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: rollout exception: {}".format(e))

    actions = np.array(actions)      # (T, 7)
    eef_poses = np.array(eef_poses)  # (T, 7)

    return success, actions, eef_poses, rgb_frames


def inject_blackout(actions, eef_poses, rgb_frames):
    """
    Pick a random blackout frame at 20-30% into the trajectory.

    Returns:
        blackout_idx (int): the step index where blackout occurs
        frozen_rgb (np.ndarray): RGB frame at blackout, uint8 (H, W, 3)
        eef_at_blackout (np.ndarray): EEF pose (7,) at blackout frame
        remaining_actions (np.ndarray): actions from blackout_idx -> end, shape (T_remain, 7)
        remaining_eef_poses (np.ndarray): EEF poses from blackout_idx -> end, shape (T_remain, 7)
    """
    T = len(actions)
    # pick blackout index at 20-30% of trajectory length
    low = max(1, int(0.20 * T))
    high = max(low + 1, int(0.30 * T))
    blackout_idx = np.random.randint(low, high)

    frozen_rgb = rgb_frames[blackout_idx]           # (H, W, 3)
    eef_at_blackout = eef_poses[blackout_idx]       # (7,)
    remaining_actions = actions[blackout_idx:]      # (T_remain, 7)
    remaining_eef_poses = eef_poses[blackout_idx:]  # (T_remain, 7)

    return blackout_idx, frozen_rgb, eef_at_blackout, remaining_actions, remaining_eef_poses


def collect_trajectories(args):
    # device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # load policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=True
    )

    # read horizon from checkpoint if not provided
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    # create environment — offscreen rendering required to capture RGB frames
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=True,
        verbose=True,
    )

    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    camera_name = args.camera_names[0]

    # open output HDF5
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    out_file = h5py.File(args.output_path, "w")
    grp = out_file.create_group("data")

    n_success = 0
    n_attempts = 0
    print("Collecting {} successful trajectories...".format(args.n_success))

    while n_success < args.n_success:
        n_attempts += 1
        print("Attempt {} | Collected so far: {}".format(n_attempts, n_success))

        success, actions, eef_poses, rgb_frames = rollout_with_obs(
            policy, env, rollout_horizon, camera_name=camera_name
        )

        if not success:
            print("  -> Failed, skipping.")
            continue

        T = len(actions)
        if T < 5:
            # trajectory too short to inject a meaningful blackout
            print("  -> Success but trajectory too short (T={}), skipping.".format(T))
            continue

        print("  -> Success! Trajectory length: {}".format(T))

        # inject blackout
        blackout_idx, frozen_rgb, eef_at_blackout, remaining_actions, remaining_eef_poses = \
            inject_blackout(actions, eef_poses, rgb_frames)

        print("  -> Blackout at step {} ({:.1f}% of trajectory)".format(
            blackout_idx, 100.0 * blackout_idx / T
        ))
        print("  -> Remaining steps after blackout: {}".format(len(remaining_actions)))

        # save to HDF5
        ep_grp = grp.create_group("demo_{}".format(n_success))

        # frozen RGB frame at blackout — this is what the human will see
        ep_grp.create_dataset("frozen_rgb", data=frozen_rgb)           # (H, W, 3) uint8

        # EEF pose at blackout frame
        ep_grp.create_dataset("eef_at_blackout", data=eef_at_blackout) # (7,) float64

        # full action sequence and EEF poses from blackout -> end (ground truth)
        ep_grp.create_dataset("remaining_actions", data=remaining_actions)     # (T_remain, 7)
        ep_grp.create_dataset("remaining_eef_poses", data=remaining_eef_poses) # (T_remain, 7)

        # metadata
        ep_grp.attrs["blackout_idx"] = blackout_idx
        ep_grp.attrs["total_traj_len"] = T
        ep_grp.attrs["remaining_len"] = len(remaining_actions)
        ep_grp.attrs["human_recorded"] = False  # Script 2 will flip this to True

        n_success += 1
        print("  -> Saved as demo_{}. Total collected: {}/{}".format(
            n_success - 1, n_success, args.n_success
        ))

    # global metadata
    grp.attrs["n_demos"] = n_success
    grp.attrs["camera_name"] = camera_name
    out_file.close()

    print("\nDone. {} successful trajectories saved to: {}".format(n_success, args.output_path))
    print("Attempts needed: {}".format(n_attempts))
    print("\nNext step: copy {} to your Mac and run Script 2.".format(args.output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )
    parser.add_argument(
        "--n_success",
        type=int,
        default=50,
        help="number of successful trajectories to collect",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="robot_trajectories.hdf5",
        help="path to output HDF5 file",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="(optional) override rollout horizon from checkpoint",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="(optional) override env name from checkpoint",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="camera to use for rendering RGB frames",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="(optional) random seed",
    )

    args = parser.parse_args()
    collect_trajectories(args)
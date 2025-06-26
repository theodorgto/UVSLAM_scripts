#!/usr/bin/env python3
"""
Plot trajectory x, y, z vs time along with depth vs time.

Usage:
    python plot_traj.py --traj traj_imu.txt --depth depth.csv [--offset 0.0]

Options:
    --traj / -t     Path to trajectory file in TUM format (each row: timestamp x y z qx qy qz qw).
    --depth / -d    Path to depth CSV with columns 'timestamp' and 'value', or a two-column CSV (first=timestamp, second=depth).
    --offset        Optional float: add this to all trajectory timestamps to align with depth timestamps (default 0.0).
    --no-mark       If set, do not mark interpolated depth at traj times on the depth plot.
    --output / -o   If provided, save the figure to this filename (e.g. "out.png") instead of showing interactively.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Plot x,y,z vs time and depth vs time")
    parser.add_argument("--traj", "-t", required=True,
                        help="Trajectory file in TUM format: timestamp x y z qx qy qz qw per row")
    parser.add_argument("--depth", "-d", required=True,
                        help="Depth CSV: columns 'timestamp' and 'value', or two columns without header")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Time offset to add to trajectory timestamps (default 0.0)")
    parser.add_argument("--no-mark", action="store_true",
                        help="Do not mark interpolated depth at traj times on depth plot")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="If set, save the figure to this file instead of showing")
    return parser.parse_args()

def read_trajectory(path):
    """
    Read trajectory in TUM format: each row has at least 4 columns: t, x, y, z, ...
    Returns:
      times: np.ndarray of shape (N,)
      positions: np.ndarray of shape (N,3) for x,y,z
    """
    try:
        data = np.loadtxt(path)
    except Exception as e:
        raise RuntimeError(f"Could not read trajectory file '{path}': {e}")
    if data.ndim == 1:
        # single line
        if data.size < 4:
            raise ValueError(f"Trajectory file '{path}' has fewer than 4 numbers in its single row.")
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Trajectory file '{path}' must have at least 4 columns per row: t x y z ...")
    times = data[:, 0].astype(float) * 1e9
    positions = data[:, 1:4].astype(float)
    return times, positions

def read_depth(path):
    """
    Read depth CSV: expects columns 'timestamp' and 'value'. If not present, assumes first two columns are timestamp and depth.
    Returns:
      df: pandas.DataFrame sorted by timestamp, columns ['timestamp','value']
      times: np.ndarray of timestamps
      vals: np.ndarray of depth values
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Could not read depth CSV '{path}': {e}")
    if 'timestamp' in df.columns and 'value' in df.columns:
        df2 = df[['timestamp', 'value']].copy()
    else:
        if df.shape[1] < 2:
            raise ValueError(f"Depth CSV '{path}' must have at least two columns.")
        df2 = df.iloc[:, :2].copy()
        df2.columns = ['timestamp', 'value']
        print("Warning: depth CSV did not have 'timestamp'/'value' columns; assuming first col=timestamp, second=depth.")
    # Drop NaNs and convert
    df2 = df2.dropna(subset=['timestamp', 'value'])
    df2['timestamp'] = pd.to_numeric(df2['timestamp'], errors='coerce')
    df2['value'] = pd.to_numeric(df2['value'], errors='coerce')
    df2 = df2.dropna(subset=['timestamp', 'value'])
    # Sort ascending
    df2 = df2.sort_values(by='timestamp').reset_index(drop=True)
    times = df2['timestamp'].to_numpy(dtype=float)
    vals = df2['value'].to_numpy(dtype=float)
    return df2, times, vals

def interpolate_depth(traj_times, depth_times, depth_vals):
    """
    Linearly interpolate depth_vals at traj_times.
    For traj_times outside [min(depth_times), max(depth_times)], returns np.nan.
    """
    # np.interp with left/right set to nan (numpy>=1.10)
    depth_interp = np.interp(traj_times, depth_times, depth_vals, left=np.nan, right=np.nan)
    return depth_interp

def main():
    args = parse_args()
    # Read traj
    traj_times, positions = read_trajectory(args.traj)
    # Apply offset
    if args.offset != 0.0:
        traj_times = traj_times + args.offset

    # Read depth
    df_depth, depth_times, depth_vals = read_depth(args.depth)
    if len(depth_times) == 0:
        raise ValueError("No valid depth data found in the CSV.")

    # Interpolate depth at traj times (for marking)
    depth_on_traj = interpolate_depth(traj_times, depth_times, depth_vals)

    # Create 4 subplots stacked, shared x-axis
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8))
    ax_x, ax_y, ax_z, ax_mag, ax_d = axes

    # calculate the resulting length of the trajectory using pythagorean theorem
    traj_magnitude = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)

    # Plot x vs time
    ax_x.plot(traj_times, positions[:, 0], '-o', markersize=3, label='x')
    ax_x.set_ylabel("X")
    ax_x.set_title("Trajectory x, y, z vs Time & Depth vs Time")
    ax_x.grid(True)

    # Plot y vs time
    ax_y.plot(traj_times, positions[:, 1], '-o', markersize=3, label='y', color='tab:orange')
    ax_y.set_ylabel("Y")
    ax_y.grid(True)

    # Plot z vs time
    ax_z.plot(traj_times, positions[:, 2], '-o', markersize=3, label='z', color='tab:green')
    ax_z.set_ylabel("Z")
    ax_z.grid(True)

    # Plot magnitude vs time
    ax_mag.plot(traj_times, -traj_magnitude, '-o', markersize=3, label='magnitude', color='tab:red')
    ax_mag.set_ylabel("Magnitude")
    ax_mag.set_xlabel("Time")
    ax_mag.grid(True)

    # Plot depth vs time
    ax_d.plot(depth_times, depth_vals, '-', linewidth=1, label='depth')
    ax_d.plot(traj_times, -traj_magnitude/5, '--', linewidth=1, label='traj magnitude', color='tab:green')
    ax_d.set_ylabel("Depth")
    ax_d.set_xlabel("Time")
    ax_d.grid(True)
    # Mark interpolated depth at traj times, if not disabled
    if not args.no_mark:
        # Only mark where depth_on_traj is not nan
        valid = ~np.isnan(depth_on_traj)
        if np.any(valid):
            ax_d.scatter(traj_times[valid], depth_on_traj[valid],
                         marker='x', color='red', s=30, label='depth@traj times')
        # Optionally, mark times without depth as special marker
        invalid = np.isnan(depth_on_traj)
        if np.any(invalid):
            # Place them at bottom of plot area
            ymin, ymax = ax_d.get_ylim()
            y_mark = ymin + 0.02*(ymax-ymin)
            ax_d.scatter(traj_times[invalid], np.full_like(traj_times[invalid], y_mark),
                         marker='|', color='gray', s=20, label='traj w/o depth')
        ax_d.legend(loc='best', fontsize='small')

    plt.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=200)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()

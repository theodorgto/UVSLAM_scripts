#!/usr/bin/python3
import pandas as pd
import numpy as np
import argparse

def load_imu_csv(path):
    """
    Load IMU CSV. Expects a timestamp column and numeric measurement columns.
    """
    df = pd.read_csv(path)
    return df

def interpolate_imu(df, timestamp_col, new_rate_hz=None, factor=None):
    """
    Interpolate IMU data to a higher rate.
    
    - df: DataFrame with timestamp_col and other numeric columns.
    - timestamp_col: name of timestamp column in df.
    - new_rate_hz: desired output rate in Hz (optional).
    - factor: multiply original rate by this factor (optional).
    
    Exactly one of new_rate_hz or factor must be provided.
    
    Returns a new DataFrame with:
    - timestamps in same units as original.
    - interpolated numeric columns.
    """
    # 1. Ensure timestamp is numeric
    ts = df[timestamp_col].values
    # Determine dtype: if integers (e.g. nanoseconds), convert to float seconds
    # Here we keep original units but compute differences in same units.
    # We'll create new timestamps spaced uniformly between min and max.
    
    # Compute original time spacing: approximate sampling rate
    # Use median diff
    diffs = np.diff(ts)
    if len(diffs) == 0:
        raise ValueError("IMU data has fewer than 2 samples")
    median_dt = np.median(diffs)
    orig_rate = 1.0 / median_dt if median_dt != 0 else None
    
    print(f"Original IMU median dt = {median_dt} (units), approx rate = {orig_rate} (1/units)")
    
    # Build new timestamp array
    t_min = ts[0]
    t_max = ts[-1]
    
    if new_rate_hz is not None:
        # Need to convert new_rate_hz (Hz) into time units matching ts.
        # If ts in seconds: new_dt = 1/new_rate_hz. If ts in nanoseconds: new_dt = 1e9/new_rate_hz.
        # Heuristic: if ts dtype large ints, assume nanoseconds; else seconds.
        if np.issubdtype(ts.dtype, np.integer):
            # assume nanoseconds
            new_dt = int(1e9 / new_rate_hz)
        else:
            # float seconds
            new_dt = 1.0 / new_rate_hz
    else:
        # factor given: target dt = median_dt / factor
        target_dt = median_dt / factor
        if np.issubdtype(ts.dtype, np.integer):
            # keep integer dtype for timestamps: round to nearest int
            new_dt = int(round(target_dt))
        else:
            new_dt = target_dt
    
    # Build new timestamp vector from t_min to t_max inclusive
    # Ensure correct dtype
    if np.issubdtype(ts.dtype, np.integer):
        new_ts = np.arange(t_min, t_max + 1, new_dt, dtype=np.int64)
    else:
        new_ts = np.arange(t_min, t_max, new_dt, dtype=np.float64)
    
    print(f"Interpolating to {len(new_ts)} points, dt = {new_dt}")
    
    # 2. Prepare DataFrame for interpolation: set index to timestamp
    df2 = df.set_index(timestamp_col)
    
    # For pandas interpolation, convert index to numeric type
    df2 = df2.sort_index()
    
    # Reindex with new timestamps
    # Create new DataFrame with index=new_ts, then combine
    df_new = pd.DataFrame(index=new_ts)
    # Join: align on index
    df_interp = df2.reindex(df2.index.union(df_new.index)).sort_index().interpolate(method='linear').reindex(new_ts)
    
    # Reset index to column
    df_interp = df_interp.reset_index().rename(columns={'index': timestamp_col})
    return df_interp

def main():
    parser = argparse.ArgumentParser(
        description="Upsample IMU CSV by interpolation"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to original IMU CSV"
    )
    parser.add_argument(
        "--output_csv", type=str, required=True,
        help="Path to write interpolated IMU CSV"
    )
    parser.add_argument(
        "--timestamp_col", type=str, default="timestamp",
        help="Name of timestamp column in the CSV"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--new_rate_hz", type=float,
        help="Desired output IMU rate in Hz (e.g. 300 for 10x 30Hz)"
    )
    group.add_argument(
        "--factor", type=float,
        help="Multiply original IMU rate by this factor"
    )
    args = parser.parse_args()
    
    df = load_imu_csv(args.input_csv)
    df_interp = interpolate_imu(df, args.timestamp_col, new_rate_hz=args.new_rate_hz, factor=args.factor)
    df_interp.to_csv(args.output_csv, index=False)
    print(f"Interpolated IMU data saved to {args.output_csv}. "
          f"Original rows: {len(df)}, New rows: {len(df_interp)}")

if __name__ == "__main__":
    main()

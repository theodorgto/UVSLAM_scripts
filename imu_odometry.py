import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

def estimate_odometry(df, start_ns, end_ns):
    # Filter by time window
    mask = (df['timestamp'] >= start_ns) & (df['timestamp'] <= end_ns)
    data = df.loc[mask].reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("Not enough data in the specified time range.")

    # Initialize state
    positions = []
    velocities = []
    orientations = []

    p = np.zeros(3)  # initial position
    v = np.zeros(3)  # initial velocity
    q = R.from_quat(df.loc[0, ['orientation_x','orientation_y','orientation_z','orientation_w']].values)

    positions.append(p.copy())
    velocities.append(v.copy())
    orientations.append(q.as_quat())

    # Loop through IMU samples
    for i in range(1, len(data)):
        dt = (data.loc[i, 'timestamp'] - data.loc[i-1, 'timestamp']) * 1e-9  # ns to s

        # Update orientation by integrating angular velocity
        omega = data.loc[i-1, ['angular_vel_x','angular_vel_y','angular_vel_z']].values
        # small-angle approximation
        dq = R.from_rotvec(omega * dt)
        q = q * dq

        # Rotate acceleration into world frame and subtract gravity
        acc_body = data.loc[i-1, ['linear_acc_x','linear_acc_y','linear_acc_z']].values
        acc_world = q.apply(acc_body)
        print(f"total acc_body : {np.linalg.norm(acc_body):.6f} m/s², acc_world: {np.linalg.norm(acc_world):.6f} m/s²")
        print(f"dt: {dt:.6f} s, acc_body: {acc_body}, acc_world: {acc_world}")
        g = np.array([0, 0, 9.80665])
        acc_world -= g
        print(f"acc_world after gravity subtraction: {acc_world}")

        # Integrate velocity and position (trapezoidal)
        v = v + acc_world * dt
        p = p + v * dt + 0.5 * acc_world * dt**2

        positions.append(p.copy())
        velocities.append(v.copy())
        orientations.append(q.as_quat())

    # Pack into DataFrame
    result = pd.DataFrame({
        'timestamp': data['timestamp'],
        'pos_x': [p[0] for p in positions],
        'pos_y': [p[1] for p in positions],
        'pos_z': [p[2] for p in positions],
        'vel_x': [v[0] for v in velocities],
        'vel_y': [v[1] for v in velocities],
        'vel_z': [v[2] for v in velocities],
        'ori_x': [q[0] for q in orientations],
        'ori_y': [q[1] for q in orientations],
        'ori_z': [q[2] for q in orientations],
        'ori_w': [q[3] for q in orientations],
            # raw IMU acc data
        'linear_acc_x': data['linear_acc_x'],
        'linear_acc_y': data['linear_acc_y'],
        'linear_acc_z': data['linear_acc_z'],
        'angular_vel_x': data['angular_vel_x'],
    })
    return result


def main():
    parser = argparse.ArgumentParser(description="IMU-based Odometry Estimation")
    parser.add_argument('csv_file', help='Path to IMU CSV file')
    # parser.add_argument('--start', type=int, required=True,
    #                     help='Start timestamp in nanoseconds')
    # parser.add_argument('--end', type=int, required=True,
    #                     help='End timestamp in nanoseconds')
    # parser.add_argument('--output', default='odometry.csv',
    #                     help='Output CSV for estimated odometry')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv_file)

    print(df)

    start_ns = 17423876_00924624256
    end_ns =   17423876_01927688408
    # end_ns = 17423876_60927688408

    output_file = 'odometry_est.csv'


    # Estimate odometry
    odo = estimate_odometry(df, start_ns, end_ns)
    odo.to_csv(output_file, index=False)
    print(f"Odometry saved to {output_file}")

if __name__ == '__main__':
    main()

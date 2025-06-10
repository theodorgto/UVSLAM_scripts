import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_odometry(csv_file, show_velocity=False, show_acceleration=False):
    # Load the odometry estimates
    df = pd.read_csv(csv_file)
    
    # Convert timestamps from nanoseconds to seconds relative to start
    t0 = df['timestamp'].iloc[0]
    df['t_sec'] = (df['timestamp'] - t0) * 1e-9
    
    # Plot position
    plt.figure(figsize=(10, 6))
    plt.plot(df['t_sec'], df['pos_x'], label='pos_x')
    plt.plot(df['t_sec'], df['pos_y'], label='pos_y')
    plt.plot(df['t_sec'], df['pos_z'], label='pos_z')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Odometry Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig('odometry_position.png')

    if show_velocity:
        # Plot velocity if requested
        plt.figure(figsize=(10, 6))
        plt.plot(df['t_sec'], df['vel_x'], label='vel_x')
        plt.plot(df['t_sec'], df['vel_y'], label='vel_y')
        plt.plot(df['t_sec'], df['vel_z'], label='vel_z')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Odometry Velocity vs Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig('odometry_velocity.png')
    
    if show_acceleration:
        # Plot acceleration if requested (not implemented yet)
        plt.figure(figsize=(10, 6))
        plt.plot(df['t_sec'], df['linear_acc_x'], label='linear_acc_x')
        plt.plot(df['t_sec'], df['linear_acc_y'], label='linear_acc_y')
        plt.plot(df['t_sec'], df['linear_acc_z'], label='linear_acc_z')
        # also plot a line at y=gravity
        plt.axhline(y=9.81, color='r', linestyle='--', label='Gravity (m/s²)')
        plt.xlabel('Time (s)')
        # set grid size in y to 1
        plt.yticks(range(-2,11,1))
        plt.ylabel('Acceleration (m/s²)')
        plt.title('Odometry Acceleration vs Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig('odometry_acceleration.png')
        
def main():
    parser = argparse.ArgumentParser(description="Visualize IMU‐based odometry over time")
    parser.add_argument('odo_csv', help='CSV file with odometry (timestamp, pos_*, vel_*, ori_*)')
    parser.add_argument('--velocity', action='store_true',
                        help='Also plot velocity components')
    parser.add_argument('--acceleration', action='store_true',
                        help='Also plot acceleration components (not implemented yet)')
    args = parser.parse_args()

    plot_odometry(args.odo_csv, show_velocity=args.velocity, show_acceleration=args.acceleration)

if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import csv

# Function to read and preprocess magnetic sensor data from multiple files
def read_magnetic_data(file_paths):
    timestamps = []
    sensors_data = {i: {'x': [], 'y': [], 'z': []} for i in range(1, 6)}
    averages = []

    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = [list(map(float, line.split())) for line in file]
            data = np.array(data)

            # Print the first few lines to debug the structure
            print(f"First few lines of data from {file_path}:")
            print(data[:5])

            # Ensure we only use the first 5 columns (ignoring temperature)
            data = data[:, :5]

            # Calculate the mean of the first 10 rows for each sensor
            mean_values = {i: {'x': 0, 'y': 0, 'z': 0} for i in range(1, 6)}
            for i in range(1, 6):
                sensor_data = data[data[:, 1] == i]
                if len(sensor_data) > 10:
                    mean_values[i]['x'] = np.mean(sensor_data[:10, 2])
                    mean_values[i]['y'] = np.mean(sensor_data[:10, 3])
                    mean_values[i]['z'] = np.mean(sensor_data[:10, 4])

            # Subtract the mean values from the remaining data for each sensor
            for row in data:
                sensor_id = int(row[1])
                row[2] -= mean_values[sensor_id]['x']
                row[3] -= mean_values[sensor_id]['y']
                row[4] -= mean_values[sensor_id]['z']

            # Store the adjusted data
            for row in data:
                timestamp, sensor_id, x_value, y_value, z_value = row
                timestamp = float(timestamp)
                sensor_id = int(sensor_id)
                if sensor_id == 1:
                    timestamps.append(timestamp)
                sensors_data[sensor_id]['x'].append(x_value)
                sensors_data[sensor_id]['y'].append(y_value)
                sensors_data[sensor_id]['z'].append(z_value)

            # Save the average values for this file
            averages.append([file_path] + [mean_values[i]['x'] for i in range(1, 6)] +
                            [mean_values[i]['y'] for i in range(1, 6)] + [mean_values[i]['z'] for i in range(1, 6)])

    return timestamps, sensors_data, averages

# The rest of the functions remain the same

# Function to read and preprocess CNC log data from multiple files
def read_cnc_data(file_paths):
    timestamps = []
    x_values = []
    y_values = []
    z_values = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = [list(map(float, line.split())) for line in file if float(line.split()[3]) < 0]
            data = np.array(data)
            if len(data) > 10:
                # Calculate the mean of the first 10 rows for each column
                mean_values = np.mean(data[:10, 1:], axis=0)
                # Subtract the mean values from the remaining data
                data[:, 1:] -= mean_values
            for row in data:
                timestamp, x_value, y_value, z_value = row
                timestamps.append(timestamp)
                x_values.append(x_value)
                y_values.append(y_value)
                z_values.append(z_value)
    return timestamps, x_values, y_values, z_values

# Function to read and preprocess force torque sensor data from multiple files
def read_force_torque_data(file_paths):
    timestamps = []
    force_torque_data = {'tx': [], 'ty': [], 'tz': [], 'fx': [], 'fy': [], 'fz': []}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = [list(map(float, line.split())) for line in file]
            data = np.array(data)
            if len(data) > 10:
                # Calculate the mean of the first 10 rows for each column
                mean_values = np.mean(data[:10, 2:], axis=0)
                # Subtract the mean values from the remaining data
                data[:, 2:] -= mean_values
            for row in data:
                timestamp, _, fx, fy, fz, tx, ty, tz = row
                timestamps.append(timestamp)
                force_torque_data['tx'].append(tx)
                force_torque_data['ty'].append(ty)
                force_torque_data['tz'].append(tz)
                force_torque_data['fx'].append(fx)
                force_torque_data['fy'].append(fy)
                force_torque_data['fz'].append(fz)
    return timestamps, force_torque_data

# Normalize timestamps based on the earliest start time
def normalize_timestamps(timestamps, start_time):
    return [t - start_time for t in timestamps]

# Find the closest index in a sorted array
def find_closest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return idx - 1
    else:
        return idx

# Main function to execute the data processing and plotting
def main():
    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt the user to select the magnetic sensor data files
    magnetic_file_paths = filedialog.askopenfilenames(title="Select Magnetic Sensor Data Files", filetypes=[("Text files", "*.txt")])
    # Read magnetic sensor data
    timestamps_magnetic, sensors_data, averages = read_magnetic_data(magnetic_file_paths)

    # Prompt the user to select the CNC log data files
    cnc_file_paths = filedialog.askopenfilenames(title="Select CNC Log Data Files", filetypes=[("Text files", "*.txt")])
    # Read CNC log data
    timestamps_cnc, x_values_cnc, y_values_cnc, z_values_cnc = read_cnc_data(cnc_file_paths)

    # Prompt the user to select the force torque sensor data files
    force_torque_file_paths = filedialog.askopenfilenames(title="Select Force Torque Sensor Data Files", filetypes=[("Text files", "*.txt")])
    # Read force torque sensor data
    timestamps_force_torque, force_torque_data = read_force_torque_data(force_torque_file_paths)

    # Find the earliest timestamp from all datasets
    start_time = min(timestamps_magnetic[0], timestamps_cnc[0], timestamps_force_torque[0])

    # Normalize timestamps based on the earliest start time
    timestamps_magnetic = normalize_timestamps(timestamps_magnetic, start_time)
    timestamps_cnc = normalize_timestamps(timestamps_cnc, start_time)
    timestamps_force_torque = normalize_timestamps(timestamps_force_torque, start_time)

    # Match magnetometer and force torque data points to CNC data points based on timestamps and include 0.5 to 1.5 seconds data
    combined_data = []

    for i, t_cnc in enumerate(timestamps_cnc):
        closest_idx_mag = find_closest_index(timestamps_magnetic, t_cnc)
        closest_idx_ft = find_closest_index(timestamps_force_torque, t_cnc)

        start_time_window = t_cnc + 0.5
        end_time_window = t_cnc + 1.5

        mag_data_points = {1: [], 2: [], 3: [], 4: [], 5: []}
        ft_data_points = {'tx': [], 'ty': [], 'tz': [], 'fx': [], 'fy': [], 'fz': []}

        while closest_idx_mag < len(timestamps_magnetic) and timestamps_magnetic[closest_idx_mag] <= end_time_window:
            if timestamps_magnetic[closest_idx_mag] >= start_time_window:
                for sensor_id in mag_data_points:
                    mag_data_points[sensor_id].append([
                        sensors_data[sensor_id]['x'][closest_idx_mag],
                        sensors_data[sensor_id]['y'][closest_idx_mag],
                        sensors_data[sensor_id]['z'][closest_idx_mag]
                    ])
            closest_idx_mag += 1

        while closest_idx_ft < len(timestamps_force_torque) and timestamps_force_torque[closest_idx_ft] <= end_time_window:
            if timestamps_force_torque[closest_idx_ft] >= start_time_window:
                ft_data_points['tx'].append(force_torque_data['tx'][closest_idx_ft])
                ft_data_points['ty'].append(force_torque_data['ty'][closest_idx_ft])
                ft_data_points['tz'].append(force_torque_data['tz'][closest_idx_ft])
                ft_data_points['fx'].append(force_torque_data['fx'][closest_idx_ft])
                ft_data_points['fy'].append(force_torque_data['fy'][closest_idx_ft])
                ft_data_points['fz'].append(force_torque_data['fz'][closest_idx_ft])
            closest_idx_ft += 1

        if mag_data_points[1] and ft_data_points['tx']:
            avg_mag_data = {
                sensor_id: np.mean(mag_data_points[sensor_id], axis=0) for sensor_id in mag_data_points
            }
            avg_ft_data = {key: np.mean(ft_data_points[key]) for key in ft_data_points}

            combined_data.append([
                t_cnc, x_values_cnc[i], y_values_cnc[i], z_values_cnc[i],
                start_time_window,  # representative timestamp for the window
                avg_mag_data[1][0], avg_mag_data[1][1], avg_mag_data[1][2],
                avg_mag_data[2][0], avg_mag_data[2][1], avg_mag_data[2][2],
                avg_mag_data[3][0], avg_mag_data[3][1], avg_mag_data[3][2],
                avg_mag_data[4][0], avg_mag_data[4][1], avg_mag_data[4][2],
                avg_mag_data[5][0], avg_mag_data[5][1], avg_mag_data[5][2],
                start_time_window,  # representative timestamp for the window
                avg_ft_data['tx'], avg_ft_data['ty'], avg_ft_data['tz'],
                avg_ft_data['fx'], avg_ft_data['fy'], avg_ft_data['fz']
            ])

    combined_data = np.array(combined_data)

    # Print the combined matrix shape and a few rows for verification
    print("Combined data shape:", combined_data.shape)
    print("First 5 rows of combined data:\n", combined_data[:5])

    # Generate the current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the combined matrix to a file for further use
    np.savetxt(f'combined_data_{current_time}.csv', combined_data, delimiter=',', header='timestamp,x_cnc,y_cnc,z_cnc,timestamp_mag,Bx1,By1,Bz1,Bx2,By2,Bz2,Bx3,By3,Bz3,Bx4,By4,Bz4,Bx5,By5,Bz5,timestamp_ft,tx,ty,tz,fx,fy,fz', comments='')

    # Save the averages to a separate file using csv module
    with open(f'averages_{current_time}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['file_path', 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 'Bx3', 'By3', 'Bz3', 'Bx4', 'By4', 'Bz4', 'Bx5', 'By5', 'Bz5'])
        for avg in averages:
            csvwriter.writerow(avg)

if __name__ == "__main__":
    main()

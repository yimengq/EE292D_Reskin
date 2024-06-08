import numpy as np
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import csv
import re
from tqdm import tqdm
import time
from scipy.spatial import cKDTree

# Function to read and preprocess magnetic sensor data from a file
def read_magnetic_data(file_path):
    timestamps = []
    sensors_data = {i: {'x': [], 'y': [], 'z': []} for i in range(1, 6)}
    averages = []

    with open(file_path, 'r') as file:
        data = [list(map(float, line.split())) for line in file]
        data = np.array(data)

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

# Function to read and preprocess CNC log data from a file
def read_cnc_data(file_path):
    timestamps = []
    x_values = []
    y_values = []
    z_values = []
    with open(file_path, 'r') as file:
        data = [list(map(float, line.split())) for line in file]
        data = np.array(data[:-3])
        for row in data:
            timestamp, x_value, y_value, z_value = row
            timestamps.append(timestamp)
            x_values.append(x_value)
            y_values.append(y_value)
            z_values.append(z_value)
    return timestamps, x_values, y_values, z_values

# Function to read and preprocess force torque sensor data from a file
def read_force_torque_data(file_path):
    timestamps = []
    force_torque_data = {'tx': [], 'ty': [], 'tz': [], 'fx': [], 'fy': [], 'fz': []}
    with open(file_path, 'r') as file:
        data = [list(map(float, line.split())) for line in file]
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
def find_closest_index1(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return idx - 1
    else:
        return idx
    
def find_closest_index(kdtree, value):
    distance, idx = kdtree.query(np.array([value]).reshape(-1, 1))
    return idx[0]

# Main function to execute the data processing and plotting
def main(reduction=False, calib_window_length=60):
    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt the user to select the magnetic sensor data files
    magnetic_file_paths = filedialog.askopenfilenames(title="Select Magnetic Sensor Data Files", filetypes=[("Text files", "*.txt")])
    all_averages = []

    # Process each magnetometer data file separately
    for mag_file in magnetic_file_paths:
        # Read magnetic sensor data
        timestamps_magnetic, sensors_data, averages = read_magnetic_data(mag_file)
        all_averages.extend(averages)

        # Extract the original timestamp from the file name
        match = re.search(r'\d{8}_\d{6}', mag_file)
        if match:
            timestamp_str = match.group(0)
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prompt the user to select the corresponding CNC log data file
        cnc_file_path = filedialog.askopenfilename(title=f"Select CNC Log Data File for {mag_file}", filetypes=[("Text files", "*.txt")])
        # Read CNC log data
        timestamps_cnc_raw, x_values_cnc_raw, y_values_cnc_raw, z_values_cnc_raw = read_cnc_data(cnc_file_path)
        cnc_poke_idx  = np.where(np.array(z_values_cnc_raw)<0)[0]   # actual poking index
        cnc_calib_idx = cnc_poke_idx-1                              # calibration point index, it's the point right before the poke, no touch
        timestamps_cnc, x_values_cnc, y_values_cnc, z_values_cnc = [timestamps_cnc_raw[i] for i in cnc_poke_idx], \
                                                                   [x_values_cnc_raw[i] for i in cnc_poke_idx], \
                                                                   [y_values_cnc_raw[i] for i in cnc_poke_idx], \
                                                                   [z_values_cnc_raw[i] for i in cnc_poke_idx]
        timestamps_cnc_calib = [timestamps_cnc_raw[i] for i in cnc_calib_idx]

        # Prompt the user to select the corresponding force torque sensor data file
        force_torque_file_path = filedialog.askopenfilename(title=f"Select Force Torque Sensor Data File for {mag_file}", filetypes=[("Text files", "*.txt")])
        # Read force torque sensor data
        timestamps_force_torque, force_torque_data = read_force_torque_data(force_torque_file_path)

        # Find the earliest timestamp from all datasets
        start_time = min(timestamps_magnetic[0], timestamps_cnc[0], timestamps_force_torque[0])

        # Normalize timestamps based on the earliest start time
        timestamps_magnetic = normalize_timestamps(timestamps_magnetic, start_time)
        timestamps_cnc = normalize_timestamps(timestamps_cnc, start_time)
        timestamps_cnc_calib = normalize_timestamps(timestamps_cnc_calib, start_time)
        timestamps_force_torque = normalize_timestamps(timestamps_force_torque, start_time)

        # Build the KDTree for timestamps_magnetic
        t_mag_kdtree = cKDTree(np.array(timestamps_magnetic).reshape(-1, 1))

        # Process each CNC timestamp to determine the chosen time frame
        combined_data = []
        for t_cnc, x_cnc, y_cnc, z_cnc, t_cnc_calib in tqdm(zip(timestamps_cnc, x_values_cnc, y_values_cnc, z_values_cnc, timestamps_cnc_calib)):
            start_time_window = t_cnc + 0.5
            end_time_window = t_cnc + 1.5
            single_indent_data = []

            # Select force torque data within the chosen time frame
            chosen_force_torque_data = []
            for i, t_ft in enumerate(timestamps_force_torque):
                if start_time_window <= t_ft <= end_time_window:
                    chosen_force_torque_data.append([
                        t_ft,
                        force_torque_data['tx'][i], force_torque_data['ty'][i], force_torque_data['tz'][i],
                        force_torque_data['fx'][i], force_torque_data['fy'][i], force_torque_data['fz'][i]
                    ])
            
            calib_idx_mag = find_closest_index(t_mag_kdtree, t_cnc_calib)

            # For each row of the chosen force torque data, find the closest magnetometer data
            for ft_data in chosen_force_torque_data:
                t_ft = ft_data[0]
                closest_idx_mag = find_closest_index(t_mag_kdtree, t_ft)

                # get calibration data
                calib_range = slice(calib_idx_mag, calib_idx_mag + calib_window_length)
                calib_row_data = np.array([
                                        np.mean(sensors_data[1]['x'][calib_range]), np.mean(sensors_data[1]['y'][calib_range]), np.mean(sensors_data[1]['z'][calib_range]),
                                        np.mean(sensors_data[2]['x'][calib_range]), np.mean(sensors_data[2]['y'][calib_range]), np.mean(sensors_data[2]['z'][calib_range]),
                                        np.mean(sensors_data[3]['x'][calib_range]), np.mean(sensors_data[3]['y'][calib_range]), np.mean(sensors_data[3]['z'][calib_range]),
                                        np.mean(sensors_data[4]['x'][calib_range]), np.mean(sensors_data[4]['y'][calib_range]), np.mean(sensors_data[4]['z'][calib_range]),
                                        np.mean(sensors_data[5]['x'][calib_range]), np.mean(sensors_data[5]['y'][calib_range]), np.mean(sensors_data[5]['z'][calib_range])
                                    ])
                mag_row_data = np.array([sensors_data[1]['x'][closest_idx_mag], sensors_data[1]['y'][closest_idx_mag], sensors_data[1]['z'][closest_idx_mag],
                                        sensors_data[2]['x'][closest_idx_mag], sensors_data[2]['y'][closest_idx_mag], sensors_data[2]['z'][closest_idx_mag],
                                        sensors_data[3]['x'][closest_idx_mag], sensors_data[3]['y'][closest_idx_mag], sensors_data[3]['z'][closest_idx_mag],
                                        sensors_data[4]['x'][closest_idx_mag], sensors_data[4]['y'][closest_idx_mag], sensors_data[4]['z'][closest_idx_mag],
                                        sensors_data[5]['x'][closest_idx_mag], sensors_data[5]['y'][closest_idx_mag], sensors_data[5]['z'][closest_idx_mag]])

                mag_row_data = mag_row_data - calib_row_data

                single_indent_data.append([
                    t_cnc, x_cnc, y_cnc, z_cnc,
                    timestamps_magnetic[closest_idx_mag],  # timestamp of the magnetic sensor data
                    *mag_row_data.tolist(),
                    *ft_data
                ])

            if reduction:
                reduction_row = single_indent_data[0]
                reduction_row[5:20] = np.mean(np.array(single_indent_data)[:,5:20], axis=0)
                reduction_row[-1] = np.mean(np.array(single_indent_data)[:,-1], axis=0)
                combined_data.extend([reduction_row])
            else:
                combined_data.extend(single_indent_data)

        combined_data = np.array(combined_data, dtype=object)

        # Print the combined matrix shape and a few rows for verification
        print(f"Combined data shape for {mag_file}:", combined_data.shape)
        print("First 5 rows of combined data:\n", combined_data[:5])

        # Save the combined matrix to a file for further use
        np.savetxt(f'R_notbrian_combined_{timestamp_str}.csv', combined_data, delimiter=',', fmt='%s', header='timestamp,x_cnc,y_cnc,z_cnc,timestamp_mag,Bx1,By1,Bz1,Bx2,By2,Bz2,Bx3,By3,Bz3,Bx4,By4,Bz4,Bx5,By5,Bz5,timestamp_ft,tx,ty,tz,fx,fy,fz', comments='')

    # Save all the averages to a single file using csv module
    with open(f'averages_{timestamp_str}.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['file_path', 'Bx1', 'By1', 'Bz1', 'Bx2', 'By2', 'Bz2', 'Bx3', 'By3', 'Bz3', 'Bx4', 'By4', 'Bz4', 'Bx5', 'By5', 'Bz5'])
        for avg in all_averages:
            csvwriter.writerow(avg)

if __name__ == "__main__":
    main(reduction=True, calib_window_length=60)

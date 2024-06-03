import matplotlib.pyplot as plt
import numpy as np

# Function to read magnetic sensor data
def read_magnetic_data(file_path):
    timestamps = []
    sensors_data = {i: {'x': [], 'y': [], 'z': []} for i in range(1, 6)}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            timestamp = float(parts[0])
            sensor_id = int(parts[1])
            x_value = float(parts[2])
            y_value = float(parts[3])
            z_value = float(parts[4])
            if sensor_id == 1:
                timestamps.append(timestamp)
            sensors_data[sensor_id]['x'].append(x_value)
            sensors_data[sensor_id]['y'].append(y_value)
            sensors_data[sensor_id]['z'].append(z_value)
    return timestamps, sensors_data

# Function to read CNC log data
def read_cnc_data(file_path):
    timestamps = []
    x_values = []
    y_values = []
    z_values = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            z_value = float(parts[3])
            if z_value < 0:  # Only keep data points where the last column value is a negative non-zero value
                timestamps.append(float(parts[0]))
                x_values.append(float(parts[1]))
                y_values.append(float(parts[2]))
                z_values.append(z_value)
    return timestamps, x_values, y_values, z_values

# Function to read force torque sensor data
def read_force_torque_data(file_path):
    timestamps = []
    force_torque_data = {'fx': [], 'fy': [], 'fz': [], 'tx': [], 'ty': [], 'tz': []}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            timestamps.append(float(parts[1]))
            force_torque_data['fx'].append(float(parts[2]))
            force_torque_data['fy'].append(float(parts[3]))
            force_torque_data['fz'].append(float(parts[4]))
            force_torque_data['tx'].append(float(parts[5]))
            force_torque_data['ty'].append(float(parts[6]))
            force_torque_data['tz'].append(float(parts[7]))
    return timestamps, force_torque_data

# Normalize timestamps based on the earliest start time
def normalize_timestamps(timestamps, start_time):
    return [t - start_time for t in timestamps]

# Denoise the data using a moving average filter
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Find the closest index in a sorted array
def find_closest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return idx - 1
    else:
        return idx

# Read magnetic sensor data
magnetic_file_path = 'magnetometer_data_20240521_194804.txt'
timestamps_magnetic, sensors_data = read_magnetic_data(magnetic_file_path)

# Read CNC log data
cnc_file_path = 'cnc_log_20240521_194747.txt'
timestamps_cnc, x_values_cnc, y_values_cnc, z_values_cnc = read_cnc_data(cnc_file_path)

# Read force torque sensor data
force_torque_file_path = 'force_sensor_data_20240521_194804.txt'
timestamps_force_torque, force_torque_data = read_force_torque_data(force_torque_file_path)

# Find the earliest timestamp from all datasets
start_time = min(timestamps_magnetic[0], timestamps_cnc[0], timestamps_force_torque[0])

# Normalize timestamps based on the earliest start time
timestamps_magnetic = normalize_timestamps(timestamps_magnetic, start_time)
timestamps_cnc = normalize_timestamps(timestamps_cnc, start_time)
timestamps_force_torque = normalize_timestamps(timestamps_force_torque, start_time)

# Save the original (non-denoised) sensor data
original_sensors_data = {i: {'x': sensors_data[i]['x'][:], 'y': sensors_data[i]['y'][:], 'z': sensors_data[i]['z'][:]} for i in range(1, 6)}

# Apply moving average filter to each sensor
window_size = 25  # Adjust the window size as needed
for sensor_id in sensors_data:
    sensors_data[sensor_id]['x'] = moving_average(sensors_data[sensor_id]['x'], window_size)
    sensors_data[sensor_id]['y'] = moving_average(sensors_data[sensor_id]['y'], window_size)
    sensors_data[sensor_id]['z'] = moving_average(sensors_data[sensor_id]['z'], window_size)

# Denoise the force torque sensor data
for key in force_torque_data:
    force_torque_data[key] = moving_average(force_torque_data[key], window_size)

# Adjust timestamps to match the length of the denoised data
denoised_timestamps_magnetic = timestamps_magnetic[window_size - 1:]
denoised_timestamps_force_torque = timestamps_force_torque[window_size - 1:]

# Match magnetometer and force torque data points to CNC data points based on timestamps
matched_timestamps_magnetic = []
matched_timestamps_force_torque = []
matched_sensors_data = {i: {'x': [], 'y': [], 'z': []} for i in range(1, 6)}
matched_force_torque_data = {'fx': [], 'fy': [], 'fz': [], 'tx': [], 'ty': [], 'tz': []}

for t_cnc in timestamps_cnc:
    closest_idx_mag = find_closest_index(denoised_timestamps_magnetic, t_cnc)
    closest_idx_ft = find_closest_index(denoised_timestamps_force_torque, t_cnc)
    matched_timestamps_magnetic.append(denoised_timestamps_magnetic[closest_idx_mag])
    matched_timestamps_force_torque.append(denoised_timestamps_force_torque[closest_idx_ft])
    for sensor_id in range(1, 6):
        matched_sensors_data[sensor_id]['x'].append(sensors_data[sensor_id]['x'][closest_idx_mag])
        matched_sensors_data[sensor_id]['y'].append(sensors_data[sensor_id]['y'][closest_idx_mag])
        matched_sensors_data[sensor_id]['z'].append(sensors_data[sensor_id]['z'][closest_idx_mag])
    for key in force_torque_data:
        matched_force_torque_data[key].append(force_torque_data[key][closest_idx_ft])

# Create a combined matrix
combined_data = []

for i in range(len(timestamps_cnc)):
    row = [
        timestamps_cnc[i], x_values_cnc[i], y_values_cnc[i], z_values_cnc[i], 
        matched_timestamps_magnetic[i], matched_timestamps_force_torque[i],
        matched_sensors_data[1]['x'][i], matched_sensors_data[1]['y'][i], matched_sensors_data[1]['z'][i],
        matched_sensors_data[2]['x'][i], matched_sensors_data[2]['y'][i], matched_sensors_data[2]['z'][i],
        matched_sensors_data[3]['x'][i], matched_sensors_data[3]['y'][i], matched_sensors_data[3]['z'][i],
        matched_sensors_data[4]['x'][i], matched_sensors_data[4]['y'][i], matched_sensors_data[4]['z'][i],
        matched_sensors_data[5]['x'][i], matched_sensors_data[5]['y'][i], matched_sensors_data[5]['z'][i],
        matched_force_torque_data['fx'][i], matched_force_torque_data['fy'][i], matched_force_torque_data['fz'][i],
        matched_force_torque_data['tx'][i], matched_force_torque_data['ty'][i], matched_force_torque_data['tz'][i]
    ]
    combined_data.append(row)

combined_data = np.array(combined_data)

# Print the combined matrix shape and a few rows for verification
print("Combined data shape:", combined_data.shape)
print("First 5 rows of combined data:\n", combined_data[:5])

# Optionally, save the combined matrix to a file for further use
np.savetxt('combined_data.csv', combined_data, delimiter=',', header='timestamp,x_cnc,y_cnc,z_cnc,timestamp_mag,timestamp_ft,Bx1,By1,Bz1,Bx2,By2,Bz2,Bx3,By3,Bz3,Bx4,By4,Bz4,Bx5,By5,Bz5,fx,fy,fz,tx,ty,tz', comments='')

# Plot the original (non-denoised) magnetic sensor data
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

for sensor_id in range(1, 6):
    axes[0].plot(timestamps_magnetic, original_sensors_data[sensor_id]['x'], label=f'Sensor {sensor_id} X')
    axes[1].plot(timestamps_magnetic, original_sensors_data[sensor_id]['y'], label=f'Sensor {sensor_id} Y')
    axes[2].plot(timestamps_magnetic, original_sensors_data[sensor_id]['z'], label=f'Sensor {sensor_id} Z')

axes[0].set_ylabel('Magnetic Field (X)')
axes[0].legend()
axes[0].grid(True)

axes[1].set_ylabel('Magnetic Field (Y)')
axes[1].legend()
axes[1].grid(True)

axes[2].set_ylabel('Magnetic Field (Z)')
axes[2].legend()
axes[2].grid(True)

axes[2].set_xlabel('Time (s)')

fig.suptitle('Original (Non-Denoised) Magnetic Field Changes for All Sensors')

# Plot the denoised magnetic sensor data
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

for sensor_id in range(1, 6):
    axes[0].plot(matched_timestamps_magnetic, matched_sensors_data[sensor_id]['x'], label=f'Sensor {sensor_id} X', marker='o', linestyle='-')
    axes[1].plot(matched_timestamps_magnetic, matched_sensors_data[sensor_id]['y'], label=f'Sensor {sensor_id} Y', marker='o', linestyle='-')
    axes[2].plot(matched_timestamps_magnetic, matched_sensors_data[sensor_id]['z'], label=f'Sensor {sensor_id} Z', marker='o', linestyle='-')

axes[0].set_ylabel('Magnetic Field (X)')
axes[0].legend()
axes[0].grid(True)

axes[1].set_ylabel('Magnetic Field (Y)')
axes[1].legend()
axes[1].grid(True)

axes[2].set_ylabel('Magnetic Field (Z)')
axes[2].legend()
axes[2].grid(True)

axes[2].set_xlabel('Time (s)')

fig.suptitle('Denoised Magnetic Field Changes for All Sensors')

# Plot the CNC log data
plt.figure(figsize=(12, 6))

plt.plot(timestamps_cnc, x_values_cnc, label='CNC X direction', marker='o', linestyle='-')
plt.plot(timestamps_cnc, y_values_cnc, label='CNC Y direction', marker='o', linestyle='-')
plt.plot(timestamps_cnc, z_values_cnc, label='CNC Z direction', marker='o', linestyle='-')

plt.xlabel('Time (s)')
plt.ylabel('CNC Position (units)')
plt.title('CNC Position Change vs. Time')
plt.legend()
plt.grid(True)

# Plot the denoised force torque sensor data
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

axes[0].plot(matched_timestamps_magnetic, matched_force_torque_data['fx'], label='Force X', marker='o', linestyle='-')
axes[0].plot(matched_timestamps_magnetic, matched_force_torque_data['fy'], label='Force Y', marker='o', linestyle='-')
axes[0].plot(matched_timestamps_magnetic, matched_force_torque_data['fz'], label='Force Z', marker='o', linestyle='-')

axes[0].set_ylabel('Force (N)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(matched_timestamps_magnetic, matched_force_torque_data['tx'], label='Torque X', marker='o', linestyle='-')
axes[1].plot(matched_timestamps_magnetic, matched_force_torque_data['ty'], label='Torque Y', marker='o', linestyle='-')
axes[1].plot(matched_timestamps_magnetic, matched_force_torque_data['tz'], label='Torque Z', marker='o', linestyle='-')

axes[1].set_ylabel('Torque (Nm)')
axes[1].legend()
axes[1].grid(True)

axes[1].set_xlabel('Time (s)')

fig.suptitle('Denoised Force Torque Sensor Data')

plt.tight_layout()
plt.show()

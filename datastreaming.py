import serial
import time
import socket
import struct
import numpy as np

class StreamATI:
    def __init__(self) -> None:
        # Set the UDP IP and PORT
        UDP_IP = "192.169.1.26"
        UDP_PORT = 5005
        
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind the socket to the IP and PORT
        self.sock.bind((UDP_IP, UDP_PORT))


    def retrieve_data_point(self):
        # Receive data
        data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
        print(data)
        
        # Unpack the received bytes into floats
        # floats_array = [struct.unpack('d', data[i:i+8])[0] for i in range(0, len(data), 4)]
        
        t1_2 = struct.unpack('d', data[:8])[0]
        zeroed_forces_and_torques2 = struct.unpack('6f', data[8:32])
        floats_array = [t1_2] + list(zeroed_forces_and_torques2)
        return np.array(floats_array)

# Initialize ATI force sensor streaming
np.set_printoptions(precision=3, linewidth=200, suppress=True)
streamer = StreamATI()

# Specify the serial port and baud rate
port = "/dev/tty.usbmodem12401"  # Adjust this to match your Arduino's connection port
baudrate = 115200

# Initialize the serial connection
ser = serial.Serial(port, baudrate, timeout=1)

# List to store magnetometer data for each sensor
magnetometer_data = []
force_sensor_data = []

# Function to process and print a single line of data from the serial port
def process_data(line, line_time):
    parts = line.split("\t")
    if len(parts) == 20:  # We expect 20 items (5 sensors * 4 data points each)
        data = []  # List to store this timestep's data for all sensors
        # print("Readings:")
        for i in range(5):
            sensor_id = i + 1
            x, y, z, temp = parts[4 * i:4 * i + 4]
            sensor_data = [line_time, sensor_id, float(x), float(y), float(z), float(temp)]
            data.append(sensor_data)
            # print(f"Time: {line_time} Sensor {sensor_id} - X: {x} Y: {y} Z: {z} Temp: {temp}")
        magnetometer_data.append(data)
    else:
        print("Incomplete data received:", line)

try:
    # Read serial data continuously
    while True:
        line_time = time.time()
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()  # Decode the bytes to string
            process_data(line, line_time)
        # time.sleep(0.1)  # Sleep to reduce CPU load

        # Read force sensor data
        t1 = time.time()
        data_point = np.concatenate([[line_time], streamer.retrieve_data_point()])
        t2 = time.time()
        print(f"Time taken: {(t2-t1):.3f} seconds")
        print(data_point)
        force_sensor_data.append(data_point)

except KeyboardInterrupt:
    print("Program interrupted by the user.")
    print("Final data collected:")
    for idx, data in enumerate(magnetometer_data):
        print(f"Time Step {idx + 1}: {data}")

    # Save the collected data to a file, named by time
    save_time = time.strftime('%Y%m%d_%H%M%S')
    filename = f"./rawdata/magnetometer_data_{save_time}.txt"
    with open(filename, 'w') as file:
        for timestep in magnetometer_data:
            for sensor_data in timestep:
                file.write("\t".join(map(str, sensor_data)) + "\n")

    # Save the force sensor data to a file
    filename = f"./rawdata/force_sensor_data_{save_time}.txt"
    with open(filename, 'w') as file:
        for data_point in force_sensor_data:
            file.write("\t".join(map(str, data_point)) + "\n")

finally:
    ser.close()  # Close the serial connection when done

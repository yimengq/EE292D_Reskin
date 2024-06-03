import serial
import time
import numpy as np
from tqdm import tqdm

import socket, struct
from matplotlib import pyplot as plt

class StreamATI:
    def __init__(self) -> None:
        # Set the UDP IP and PORT
        UDP_IP = "192.169.1.2"
        UDP_PORT = 5005
        
        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Bind the socket to the IP and PORT
        self.sock.bind((UDP_IP, UDP_PORT))


    def retrieve_data_point(self):
        # Receive data
        data, addr = self.sock.recvfrom(1024)  # buffer size is 1024 bytes
        
        # Unpack the received bytes into floats
        floats_array = [struct.unpack('f', data[i:i+4])[0] for i in range(0, len(data), 4)]
        
        return np.array(floats_array)[:6]

class CNC:
    def __init__(self, maxfeedrate=1500, maxacc=200) -> None:
        # Set up serial connection (Adjust the COM port and baud rate)
        self.ser = serial.Serial('/dev/tty.usbserial-1220', 115200, timeout=1)
        time.sleep(2)  # Wait for connection to establish

        # variables
        self.maxfeedrate = maxfeedrate
        self.maxacc = maxacc

        # Setup machine variables
        # Set acceleration for X, Y, and Z axes
        print("Setting acceleration commands...")
        self.send_gcode(f"$120={self.maxacc}")  # X-axis acceleration to 100 mm/sec^2 [30]
        self.send_gcode(f"$121={self.maxacc}")  # Y-axis acceleration to 100 mm/sec^2 [30]
        self.send_gcode(f"$122={self.maxacc}")  # Z-axis acceleration to 100 mm/sec^2 [30]

        print("Setting speed commands...")
        self.send_gcode(f"$110={self.maxfeedrate}")  # X-axis speed to 2000 mm/sec^2 [1000]
        self.send_gcode(f"$111={self.maxfeedrate}")  # Y-axis speed to 2000 mm/sec^2 [1000]
        self.send_gcode(f"$112={self.maxfeedrate}")  # Z-axis speed to 2000 mm/sec^2 [600]

        print("Working")
        self.send_gcode("G90")  # Set positioning system [G90: Absolute, G91: Relative]
        self.send_gcode("G21")  # Set units to millimeters
        self.send_gcode("G92 X0 Y0 Z0")  # Homing command
        self.send_gcode("$10=3") # setup report

        self.cnc_log = []
        print("Setup complete")

    
    def print_buffer(self):
        while self.ser.in_waiting:  # While there's data waiting in the input buffer
            response = self.ser.readline().decode().strip()  # Read line
            # print("Buffer:",response)  # Print the initial messages


    def send_gcode(self, command, sync=False):
        self.print_buffer()
        self.ser.write(f"{command}\n".encode())  # Send command
        while True:
            response = self.ser.readline().decode().strip()  # Read response from CNC
            if response == "ok":
                # print('Done')
                if sync:
                    while not self.is_buffer_empty():
                        time.sleep(1e-5)
                break  # Exit loop if "ok" received
            else:
                print(f"Error: {response}")  # Print error message if any
                break

    def is_buffer_empty(self):
        self.ser.write(b'?\n')  # Request for status
        time.sleep(0.1)  # Give a slight delay to receive the response
        while self.ser.in_waiting:
            response = self.ser.readline().decode().strip()
            # print(response)
            if "Bf:15,127" in response:
                return True  # Buffer is empty
        # return False  # Buffer is not empty
    
    def cleanup(self):
        self.ser.close()

        # Save the CNC log to a file
        save_time = time.strftime('%Y%m%d_%H%M%S')
        filename = f"./rawdata/cnc_log_{save_time}.txt"
        with open(filename, 'w') as file:
            for log in self.cnc_log:
                file.write("\t".join(map(str, log)) + "\n")

    
    def goto(self, x, y, z, feedrate=None):
        if feedrate is None:
            feedrate = self.maxfeedrate
        self.send_gcode(f"G1 X{x} Y{y} Z{z} F{feedrate}", sync=True)
        self.cnc_log.append((time.time(), x, y, z))

    def roundtrip(self, x, y, z, feedrate=None):
        self.goto(x, y, z, feedrate)
        self.goto(-x, -y, -z, feedrate)

def generate_filtered_coordinates():
    
    # Generate a 9x9 grid of 2D coordinates within the range of 0 to 16 for both x and y
    x_values = np.linspace(0, 16, 9)
    y_values = np.linspace(0, 16, 9)
    coordinates = np.array(np.meshgrid(x_values, y_values)).T.reshape(-1, 2)

    # Function to remove a 2x2 block of points at each corner of a 9x9 grid
    def remove_corner_blocks(coords):
        # Convert array to a shape of (9, 9, 2) for easier manipulation
        grid_shape = coords.reshape(9, 9, 2)

        # Corners to remove: Top-left, Top-right, Bottom-left, Bottom-right
        # Indices for 2x2 block removal at each corner
        corners = [
            (slice(0, 2), slice(0, 2)),  # Top-left
            (slice(0, 2), slice(-2, None)),  # Top-right
            (slice(-2, None), slice(0, 2)),  # Bottom-left
            (slice(-2, None), slice(-2, None))  # Bottom-right
        ]

        # Masking the 2x2 corners
        for corner in corners:
            grid_shape[corner] = np.array([[None, None], [None, None]])

        # Flatten back to the original list of coordinates, filtering out None values
        filtered_coords = grid_shape.reshape(-1, 2)
        return filtered_coords[~np.isnan(filtered_coords).any(axis=1)]

    # Removing the 2x2 points at each corner
    filtered_coordinates = remove_corner_blocks(coordinates)
    return filtered_coordinates

"""
Before running the program: 
1. Set the cnc position to right on top of the bottom left screw
2. Move the router up 1mm from the control panel.
"""

if __name__ == '__main__':
    try:
        cnc = CNC(maxfeedrate=3000, maxacc=200)
        # cnc.goto(-3, -3, -1.5)
        cnc.goto(0, 0, 0)
        coords = generate_filtered_coordinates()

        # Plotting the coordinates
        plt.figure(figsize=(8, 8))
        alphas = np.linspace(1.0, 0.1, len(coords))
        plt.scatter(coords[:, 0], coords[:, 1], c='blue', alpha=alphas, label='Remaining Points')
        handle, = plt.plot([], [], 'go', markersize=10)  # Initialize an empty plot point
        plt.title('Remaining Points after Removing 2x2 Corners')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.grid(True)
        plt.legend()

        # for z in tqdm(np.arange(1.2, 2.3, 0.2)):
        for z in tqdm(np.arange(1.2, 2.3, 0.2)):
            for (x,y) in tqdm(coords):
                # print(x, y, z)
                cnc.goto(x, y, 1)
                cnc.goto(x, y, -z)

                # creat a point on the plot
                handle.set_data(x, y)
                plt.draw()
                plt.pause(0.05)  # Pause to update the plot
                
                time.sleep(2)
                cnc.goto(x, y, 1)

        cnc.goto(coords[-1,0], coords[-1,1], 8)
        cnc.goto(0, 0, 8)
        cnc.goto(0, 0, 0)
    except KeyboardInterrupt:
        print("Program interrupted. Cleaning up...")
    finally:
        cnc.cleanup()
        print("Cleanup complete. Exiting program.")
        plt.show()

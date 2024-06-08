import torch
import serial
import torch.nn as nn
import numpy as np
# import tensorflow as tf
from train_lightning import IndentDataset, ReskinModel
from torch.utils.data import DataLoader, random_split, ConcatDataset
from matplotlib import pyplot as plt
import time

class ReskinTorch():
    def __init__(self, model_path):
        self.model = ReskinModel.load_from_checkpoint(model_path, lr=0.0)

        self.model.eval()  # Set the model to evaluation mode
        self.denorm_vecB = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 
                            0.1, 0.1, 0.1, 0.1, 0.1, 
                            0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)*10
        self.denorm_vecF = np.array([1, 1, -1])
        
    # Function to perform inference with PyTorch model
    def infer(self, input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(input_tensor).cpu().numpy()
        return output[:2], output[2]

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
        return np.array(data)[:,2:-1].flatten()
    else:
        print("Incomplete data received:", line)
        return None


def main():
    # Initialize ATI force sensor streaming
    np.set_printoptions(precision=3, linewidth=200, suppress=True)

    # Specify the serial port and baud rate
    port = "/dev/tty.usbmodem12301"  # Adjust this to match your Arduino's connection port
    baudrate = 115200
    ser = serial.Serial(port, baudrate, timeout=1)

    # List to store magnetometer data for each sensor
    magnetometer_data = []
    
    # initialize reskin model
    model_path = "Logs/version_44/checkpoints/epoch=435.ckpt"
    reskin = ReskinTorch(model_path)
    average_lst = []
    average_flag = False

    Bwindow = []
    count = 0

    try:
        # Read serial data continuously
        while True:
            line_time = time.time()
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()  # Decode the bytes to string
                data = process_data(line, line_time)
                if data is not None:
                    if len(average_lst) < 1000:
                        average_lst.append(data)
                    else:
                        if not average_flag:
                            avg_value = np.array(average_lst).mean(axis=0)
                        data = data-avg_value
                        if len(Bwindow) > 200:
                            Bwindow.pop(0)
                        Bwindow.append(data)
                        count += 1
                        
                        if count % 100 == 0:
                            # average_B = np.array(Bwindow).mean(axis=0)
                            # plt.clf()
                            # plt.plot(np.arange(0,15), average_B)
                            # plt.grid()
                            # plt.ylim([-15,15])
                            location, force = reskin.infer(np.array(Bwindow).mean(axis=0))
                            # Bdata.append(data)
                            plt.gcf().clear()
                            plt.scatter(location[0], location[1], c='r', s=force**2 *30)
                            plt.xlim(-5,20)
                            plt.ylim(-10,20)
                            plt.grid()
                            plt.pause(0.05)

                            print(location, force)
                    
                    
                # if data_is_valid:
                #     location, force = reskin.infer(data)

    except KeyboardInterrupt:
        print("Program interrupted by the user.")
        print("Final data collected:")
        for idx, data in enumerate(magnetometer_data):
            print(f"Time Step {idx + 1}: {data}")

    finally:
        ser.close()  # Close the serial connection when done
        
    plt.show()


if __name__ == "__main__":
    main()
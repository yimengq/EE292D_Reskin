import numpy as np
import tflite_runtime.interpreter as tflite
import serial
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from collections import deque
import time
import statistics

# Function to perform inference with TFLite model
def infer_with_tflite_model(model_path, input_data):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # # Ensure input_data is 2-dimensional
    # input_data = np.expand_dims(input_data, axis=0) if input_data.ndim == 1 else input_data

    # Ensure input data is float32 and reshape to 2D
    input_data = input_data.astype(np.float32).reshape(1, -1)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    # denorm_output = output * denorm_vecF
    # return denorm_output

    return output[0][:2], output[0][2]

# # Function to read sensor data from the serial port and select specific indexes
# def read_sensor_data(serial_port, num_values, selected_indexes):
#     ser = serial.Serial(serial_port, baudrate=9600, timeout=1)
#     while True:
#         try:
#             line = ser.readline().decode('utf-8').strip()
#             if line:
#                 values = line.split()  # Assuming values are space-separated
#                 if len(values) >= max(selected_indexes) + 1:
#                     data = [float(values[i]) for i in selected_indexes]
#                     break
#         except ValueError:
#             pass
#     ser.close()
#     data = np.array(data, dtype=np.float32).reshape(1, -1)
#     data = data * norm_vecB  # Apply normalization
#     return data

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
    port = '/dev/ttyACM0'  # Adjust this to match your Arduino's connection port
    baudrate = 115200
    ser = serial.Serial(port, baudrate, timeout=1)

    # List to store magnetometer data for each sensor
    magnetometer_data = []
    
    # # initialize reskin model
    # model_path = "Logs/version_35/checkpoints/epoch=245.ckpt"
    # reskin = ReskinTorch(model_path)
    average_lst = []
    average_flag = False

    Bwindow = []
    count = 0
    timer = 0

    # Create the plot figure
    plt.ion()
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], c='r', s=30)
    ax.set_xlim(-5, 20)
    ax.set_ylim(-10, 20)
    ax.grid()

    location, force = np.array([0.0,0.0]), 0

    try:
        # Read serial data continuously
        while True:
            line_time = time.time()
            if ser.in_waiting > 0:
                # ff1 = time.time()
                line = ser.readline().decode('utf-8').strip()  # Decode the bytes to string
                data = process_data(line, line_time)
                # print('time:',time.time()-ff1)

                # if (data is not None) and (np.isnan(data).any() == False):

                if data is not None:
                    if len(average_lst) < 50:
                        average_lst.append(data)
                    else:
                        start_time = time.time()
                        if timer > 2:
                            average_lst = []
                            timer = 0
                            continue
    
                        # if not average_flag:
                        avg_value = np.array(average_lst).mean(axis=0)

                        data = data-avg_value

                        if len(Bwindow) > 200:
                            Bwindow.pop(0)
                        Bwindow.append(data)
                        count += 1
                        
                        if count % 10 == 0:
                            # location, force = reskin.infer(np.array(Bwindow).mean(axis=0))
                            location, force = infer_with_tflite_model("reskin_model.tflite", np.array(Bwindow).mean(axis=0))
                            sc.set_offsets([location])
                            sc.set_sizes([force**2 * 30])
                            plt.draw()
                            plt.pause(0.05)
                            print(timer,location, force)

                        timer = timer + (time.time() - start_time)

                
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


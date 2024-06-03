import socket
import struct
import numpy as np
import time

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

# Example usage
if __name__ == '__main__':
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    streamer = StreamATI()
    while True:
        t1 = time.time()
        data_point = streamer.retrieve_data_point()
        t2 = time.time()
        print(f"Time taken: {(t2-t1):.3f} seconds")
        print(data_point)
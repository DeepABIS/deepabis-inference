import io
import socket
import threading

import cv2
import numpy as np

from inference import BeeNet
import json


class InferenceThread(threading.Thread):
    def __init__(self, net, address, connection):
        threading.Thread.__init__(self)

        self.net = net
        self.connection = connection
        self.address = address

        print("Connected to %s on %s" %(self.address[0], self.address[1]))

    def run(self):
        while True:
            try:
                f = io.BytesIO()
                length = self.connection.recv(8)
                length = int.from_bytes(length, "big")
                b = 0
                while b < length:
                    to_read = length - b
                    data = self.connection.recv(1024 if to_read > 1024 else to_read)
                    if not data:
                        break
                    b += len(data)
                    f.write(data)
                f.seek(0)
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                top5 = self.net.infer_top5(img)
                response = json.dumps({
                    'top5': top5.tolist(),
                    'embedding': self.net.embedding,
                })
                self.connection.sendall(response.encode())  # send data to connected client
                self.connection.close()
            except socket.error as msg:
                print("Client disconnected")
                self.connection.close()  # close the socket connection
                break


if __name__ == '__main__':
    net = BeeNet()
    serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # create TCP socket
    serverSocket.bind(('0.0.0.0', 9042))
    print('listening...')

    while True:
        serverSocket.listen(1)  # listen for connections, max. non-accepted connections set to 1
        connection, address = serverSocket.accept()  # accept a connection (blocking)
        socketThread = InferenceThread(net, address, connection)
        print('starting a server thread')
        socketThread.start()

    serverSocket.close()

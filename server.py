import socket
import threading
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
                data = self.connection.recv(1024)  # receive data to connected client (blocking)

                if data:
                    print('Client sent: "%s"' % (data))
                    data = data.decode()
                    path = '../storage/python/predict/' + data
                    top5 = self.net.infer_top5(path)
                    response = json.dumps({
                        'top5': top5.tolist(),
                        'embedding': self.net.embedding,
                    })
                self.connection.sendall(response.encode())  # send data to connected client
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

import socket

socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip = "0.0.0.0"
port = 5005
serverAddress = (ip, port)
socket.bind(serverAddress)
socket.listen(1)
print("Waiting for connection")
connection, add = socket.accept()
data = connection.recv(2048)
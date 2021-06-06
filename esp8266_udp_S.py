import socket

ip = '127.0.0.1' 
port = 4294
message = b' hi from python'
    

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

for i in range (1, 10):
    sock.sendto(message, (ip ,port))
    i += 1

print(f'UDP target ip: {ip}')
print(f'UDP target port: {port}')
print (f'Message send: {message}')



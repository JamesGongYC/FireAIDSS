import socket
import struct
import numpy as np
import time
import math
class UDPReceiver:
  def __init__(self, port=5005):
    self.port = port
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.sock.bind(('0.0.0.0', self.port))
    self.yaws = np.zeros(4)
    self.states = np.zeros(4,10)
  def recv(self, size):
    data, addr = self.sock.recvfrom(size)
    id, x, y, z, vx, vy, vz, qw, qx, qy, qz = struct.unpack(
      'fffffffffff', data)
    return id, x, y, z, vx, vy, vz, qw, qx, qy, qz
  def GETyaw(self):
    data = self.recv(400)
    qw = data[7]
    qx = data[8]
    qy = data[9]
    qz = data[10]
    if data[0] == 1:
      self.yaws[0] = math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))/math.pi * 180
    if data[0] == 2:
      self.yaws[1] = math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))/math.pi * 180
    if data[0] == 3:
      self.yaws[2] = math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))/math.pi * 180
    if data[0] == 4:
      self.yaws[3] = math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))/math.pi * 180
    return
  def yaw(self,id):
    return self.yaws[id]
  def GETall(self):
    data = self.recv(400)
    if data[0] == 1:
      self.states[0,:] = data[1:]
    if data[0] == 2:
      self.states[1,:] = data[1:]
    if data[0] == 3:
      self.states[2,:] = data[1:]
    if data[0] == 4:
      self.states[3,:] = data[1:]
    return
  def get_state(self,id):
    return self.states[id,:]

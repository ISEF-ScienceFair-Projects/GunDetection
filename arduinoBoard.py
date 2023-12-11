import serial
def sendP(message):
    ser = serial.Serial('COM7', 9600)
    val = bytes(message,"utf-8")
    ser.write(val)

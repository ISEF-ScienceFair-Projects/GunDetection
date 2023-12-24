import serial

def sendP(message):
    ser = serial.Serial('COM7', 9600, timeout=0.5)
    possible_val = [0,1,2,3]
    if int(message) in possible_val:
        val = bytes(message,"utf-8")
        while True:
            ser.write(val)
            if str(ser.read(10).decode()) == '5':
               break


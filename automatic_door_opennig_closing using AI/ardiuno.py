from pyfirmata import Arduino, SERVO
import inspect

PORT = "COM6"
pin = 10

board = Arduino(PORT)
board.digital[pin].mode = SERVO

def rotateServo(pin, angle):
    board.digital[pin].write(angle)

def doorAutomate(val):
    if val == 0:
        rotateServo(pin, 220)
    elif val == 1:
        rotateServo(pin, 40)

try:
    while True:
        val = int(input("Enter 0 to close or 1 to open the door: "))
        doorAutomate(val)
except KeyboardInterrupt:
    board.exit()

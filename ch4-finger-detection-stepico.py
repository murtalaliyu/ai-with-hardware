from machine import Pin
import time
from sys import stdin
import uselect

# Define LED pins
red1 = Pin(20, Pin.OUT)
orange1 = Pin(19, Pin.OUT)
white1 = Pin(18, Pin.OUT)
green1 = Pin(17, Pin.OUT)
blue1 = Pin(16, Pin.OUT)

# Similarly, define the LED pins for the other hand
red2 = Pin(9, Pin.OUT)
orange2 = Pin(12, Pin.OUT)
white2 = Pin(13, Pin.OUT)
green2 = Pin(14, Pin.OUT)
blue2 = Pin(15, Pin.OUT)

left_LEDs = [red1, orange1, white1, green1, blue1]
right_LEDs = [red2, orange2, white2, green2, blue2]

hand = '0'

def turn_on_LEDs(LEDs, num_LEDs):
    num_LEDs = int(num_LEDs)
    
    if num_LEDs == 0:
        # if no finger shows up
        for leds in LEDs:
            leds.value(0)
    else:
        for i in range(num_LEDs):
            # turn on the corresponding number of LEDs
            LEDs[i].value(1)
        # turn off the rest of the LEDs
        for i in range(num_LEDs, 5, 1):
            LEDs[i].value(0)
            
while True:
    # to receive input via serial communication
    select_result = uselect.select([stdin], [], [], 0)
    
    while select_result[0]:
        # assign the first character to input_character
        input_character = stdin.read(1)
        
        # if one hand shows up
        if hand == '1':
            # reset the number of hands
            hand = '0'
            turn_on_LEDs(left_LEDs, input_character)
        # if two hands show up
        elif hand == '2':
            hand = '0'
            turn_on_LEDs(right_LEDs, input_character)
        # input the number of hands
        elif input_character == '1' or input_character == '2':
            hand = input_character
        # turn all lights on for one hand for testing
        elif input_character == 'L':
            turn_on_LEDs(left_LEDs, 5)
        elif input_character == 'R':
            turn_on_LEDs(right_LEDs, 5)
    # reset the empty input holder
    select_result = uselect.select([stdin], [], [], 0)

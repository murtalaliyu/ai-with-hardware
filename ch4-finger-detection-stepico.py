from machine import Pin
import time

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

#right_red_led.off()
#green1.off()

while True:
    red1.on()
    time.sleep(1)
    red1.off()
    orange1.on()
    time.sleep(1)
    orange1.off()
    white1.on()
    time.sleep(1)
    white1.off()
    green1.on()
    time.sleep(1)
    green1.off()
    blue1.on()
    time.sleep(1)
    blue1.off()
    red2.on()
    time.sleep(1)
    red2.off()
    orange2.on()
    time.sleep(1)
    orange2.off()
    white2.on()
    time.sleep(1)
    white2.off()
    green2.on()
    time.sleep(1)
    green2.off()
    blue2.on()
    time.sleep(1)
    blue2.off()


import RPi.GPIO as GPIO
import time
import requests
import os
import subprocess
import uuid
import time
import picamera
import pygame
from pygame.locals import *
import RPi.GPIO as GPIO

camera = picamera.PiCamera()
camera.resolution = (1920, 1080)
camera.exposure_mode = 'auto'
print(camera.exposure_mode)
print(camera.exposure_compensation)
camera.meter_mode = 'spot'

#camera.iso = 100
#camera.zoom = (0.15, 0.10, 0.7, 0.7)

counter = 0

def qc_request(image):
    try:
        image = image
        file_path = os.path.join(image)
        files = {"file": open(file_path, "rb")}
        response = requests.post("http://localhost:5005/QC", files=files, timeout=3)
        print("sent")
        print(response.content)
    except Exception as err:
        print("Error While Sending to QC Service...", err)

def button_callback(channel):
    random_uuid = str(uuid.uuid4())
    print("Random UUID:", random_uuid)
    print("Button was pushed!")
    camera.capture("/home/QC/qc_images"+ random_uuid+".jpg")
    print("qc/"+ random_uuid+".jpg")
    qc_request("/home/QC/qc_images"+ random_uuid+".jpg")

button = 0

print("Captured...")
camera.capture("/home/QC/qc_images/qc_test_image.jpg")
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
debounce_delay = 0

def button_handler(channel):
    if GPIO.input(channel) == GPIO.HIGH:
        time.sleep(debounce_delay)
        button_callback(channel)


GPIO.add_event_detect(10,GPIO.RISING,callback=button_handler)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
GPIO.cleanup()
camera.close()
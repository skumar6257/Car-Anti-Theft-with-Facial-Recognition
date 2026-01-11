# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:44:23 2024

@author: skuma
"""
##### NOTE: Uncomment all the commented section while running on Rasperry PI ####

from TelegramNotificationClass import TelegramNotificationClass
from ConfigurationClass import ConfigurationClass
from FacialRecoginitionClass import FacialRecoginitionClass
from GPIOClass import GPIOClass
from AuthenticationClass import AuthenticationClass
import cv2
import time
import os
import keyboard
# import RPi.GPIO as GPIO


#Pin Defination Starts Here ##
BUZZER_PIN = 27
MOTOR_PIN = 24
LOCK_PIN = 22
UNLOCK_PIN = 23
START_BUTTON_PIN = 17
#Pin Defination Ends Here ##

#Global Variable Intitalisation Starts Here##
camera = None
THEFT_SYSTEM_TIMER = 60
LOCATION_INTERVAL = 15
HW_MODE = False   #### To made true when executing with hadware connected
monitor_time = time.time() - 1.5
#Global Variable Intitalisation Ends Here##


## Configuration Starts Here ####
configuration = ConfigurationClass('config.json')
configuration.load_configuration_file()
config_data = configuration.config_file
telegram_bot = TelegramNotificationClass(config_data['BOT_TOKEN'], 
                                         config_data['CHAT_ID'],
                                         config_data['LAST_UPDATE_ID'])
face_class = FacialRecoginitionClass('known_user_encodings.pth','KnownUser')
face_class.load_face_model()
face_class.load_facial_encodings()
gpio = GPIOClass(HW_MODE, BUZZER_PIN, MOTOR_PIN, LOCK_PIN,UNLOCK_PIN, START_BUTTON_PIN)
authentication_class = AuthenticationClass(configuration, telegram_bot, gpio, face_class,THEFT_SYSTEM_TIMER,LOCATION_INTERVAL)
## Configuration Ends Here ####

# Callback function to handle button press
def button_callback():
    global camera,monitor_time
    if gpio.START_BUTTON_STATUS:
        if gpio.ENGINE_STATUS:
            ## No ISR, so extra time to smooth button pressing event with live fetching
            #Live InCabin Monitoring
            if not camera.isOpened() or camera is None:
                camera = cv2.VideoCapture(0)
            if (time.time() - monitor_time) > 1.5:
                #Live InCabin Monitoring
                print('Live InCabin Monitoring')
                authentication_class.live_bot_monitoring(camera)
                monitor_time = time.time()
        else:
            authentication = config_data['FACE_AUTHENTICATION']
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                print('Camera Opened')
                if authentication:
                    #start_engine_and_lock()
                    gpio.engine('on')
                    gpio.car_door_lock('lock')
                else:
                    authentication = authentication_class.start_authentication(camera)

                if authentication:
                    telegram_bot.send_telegram_message('Live InCabin Monitoring On')
                    telegram_bot.send_telegram_message("Please Enter 'Live picture' or 'Location'.")
            else:
                print('Error with Camera, Camera not opened')

    else:
        if gpio.ENGINE_STATUS:
            ##Switch Off All Car Accessories##
            if camera is not None:
                camera.release()
                cv2.destroyAllWindows()
            gpio.engine('off')
            gpio.car_door_lock('unlock')
            telegram_bot.send_telegram_message('Live InCabin Monitoring Off')
            configuration.config_file['FACE_AUTHENTICATION']=False
            configuration.write_configuration_file()
        else:
            pass

    

# Add event detection with debounce - Check if working for you for me it wasn't working as residual HIGH signal was interrupting normal flow
# if HW_MODE:
#     GPIO.add_event_detect(START_BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=300)

# Main loop
try:
    if not HW_MODE:
        print("Press 'q' to mimic Start/Stop Button and 'Ctrl +C' to terminate program")
    while True:
        if HW_MODE:
            start_time = time.time() - 2.5
            ## Done as to temporary handle sudden residual HIGH signal
            # pin_status = GPIO.input(gpio.START_BUTTON_PIN)
            # if pin_status == GPIO.HIGH:
            #     if (time.time() - start_time) >= 2.5:
            #         print('Button pressed')
            #         gpio.START_BUTTON_STATUS = not gpio.START_BUTTON_STATUS
            #         start_time=time.time()
        else:
            if keyboard.is_pressed('q'):
                print('Button pressed')
                gpio.START_BUTTON_STATUS = not gpio.START_BUTTON_STATUS
        button_callback()
except KeyboardInterrupt:
    print("Program terminated")
finally:
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
    folder_path = 'captured_frames'
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    # if HW_MODE: 
    #     GPIO.cleanup()
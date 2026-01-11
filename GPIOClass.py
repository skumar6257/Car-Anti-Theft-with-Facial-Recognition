# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:32:44 2024

@author: skuma
"""
##### NOTE: Uncomment all the commented section while running on Rasperry PI ####
import time
# import RPi.GPIO as GPIO

class GPIOClass:
    
    def __init__(self,HW_MODE,buzzer_pin,motor_pin,lock_pin,unlock_pin,start_button_pin):
        #Set hardware mode
        self.HW_MODE = HW_MODE
        #Set Pin Numbers
        self.BUZZER_PIN = buzzer_pin
        self.MOTOR_PIN = motor_pin
        self.LOCK_PIN = lock_pin
        self.UNLOCK_PIN = unlock_pin
        self.START_BUTTON_PIN = start_button_pin
        self.START_BUTTON_STATUS = False
        self.CAR_DOOR_LOCK_STATUS = True
        self.ENGINE_STATUS = False

        # if self.HW_MODE:

        #     #Set GPIO Mode
        #     GPIO.setmode(GPIO.BCM)
            
        #     #Set Pin Configurations
        #     GPIO.setup(self.BUZZER_PIN, GPIO.OUT)
        #     GPIO.setup(self.LOCK_PIN, GPIO.OUT)
        #     GPIO.setup(self.UNLOCK_PIN, GPIO.OUT)
        #     GPIO.setup(self.MOTOR_PIN, GPIO.OUT)
        #     GPIO.setup(self.START_BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        
    #Car Door Lock
    def car_door_lock(self,command):
        if command == 'lock':
            # if self.HW_MODE:
            #     #Lock the Car Door
            #     GPIO.output(self.LOCK_PIN, GPIO.LOW)
            #     time.sleep(1)
            #     GPIO.output(self.LOCK_PIN, GPIO.HIGH)
            self.CAR_DOOR_LOCK_STATUS = True
            print('Car Door Locked')
        elif command == 'unlock':
            # if self.HW_MODE:
            #     #Unlock the Car Door
            #     GPIO.output(self.UNLOCK_PIN, GPIO.LOW)
            #     time.sleep(1)
            #     GPIO.output(self.UNLOCK_PIN, GPIO.HIGH)
            self.CAR_DOOR_LOCK_STATUS = False
            print('Car Door Unlocked')
        else:
            print('Wrong Command')
            

    #Alert Buzzer
    def buzzer(self,command):
        if command == 'on':
            # if self.HW_MODE:
            #     #Start the buzzer
            #     GPIO.output(self.BUZZER_PIN, GPIO.HIGH)
            print('Buzzer ON')
        elif command == 'off':
            # if self.HW_MODE:
            #     #Stop the buzzer
            #     GPIO.output(self.BUZZER_PIN, GPIO.LOW)
            print('Buzzer OFF')
        else:
            print('Wrong Command')
            
    def engine(self,command):
        if command == 'on':
            # if self.HW_MODE:
            #     #Start the buzzer
            #     GPIO.output(self.MOTOR_PIN, GPIO.LOW)
            self.ENGINE_STATUS = True
            print('Engine Start')
        elif command == 'off':
            # if self.HW_MODE:
            #     #Stop the buzzer
            #     GPIO.output(self.MOTOR_PIN, GPIO.HIGH)
            self.ENGINE_STATUS = False
            print('Engine Off')
        else:
            print('Wrong Command')
        
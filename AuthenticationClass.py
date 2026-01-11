# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:42:15 2024

@author: skuma
"""
import time
import os
import cv2
import requests
from GPSClass import GPSClass
from collections import Counter

class AuthenticationClass:
    
    def __init__(self,configuration,telegram_bot,gpio,face_class,theft_system_time,location_interval):
        self.configuration = configuration
        self.telegram_bot = telegram_bot
        self.gpio = gpio
        self.face_class = face_class
        self.gps = GPSClass()
        self.theft_system_time = theft_system_time
        self.location_interval = location_interval
    
    def start_AntiTheft_System(self):
        print('Anti Theft')
        start = time.time()
        timer_15sec = start
        time_3sec = start
        ##Set Buzzer to ON
        buzzer_status = 1
        self.gpio.buzzer('on')
        ##Send GPS Data
        self.telegram_bot.send_telegram_message("Live Location for Theft Detected")
        message_id = None
        gps_data = self.gps.fetch_GPS_Data()
        if 'Error' in gps_data:
            self.telegram_bot.send_telegram_message("Unable to Fetch GPS Coordinates")
        else:
            try:
                response = self.telegram_bot.send_telegram_live_location(gps_data[0],gps_data[1],self.theft_system_time)
                if response['ok'] == True:
                    print('Location Sent successfully')
                    message_id = response['result']['message_id']
                else:
                    print('Location Sent failed')
            except Exception as e:
                # Print the exception message
                print(f"Error occurred while Sending Location: {e}")
            
        while time.time() - start < self.theft_system_time:
            #Send GPS data for 15 sec
            if time.time() - timer_15sec > self.location_interval:
                gps_data = self.gps.fetch_GPS_Data()
                if 'Error' in gps_data:
                    self.telegram_bot.send_telegram_message("Unable to Fetch GPS Coordinates")
                else:
                    if message_id is not None:
                        try:
                            self.telegram_bot.update_telegram_live_location(gps_data[0],gps_data[1],message_id)
                            timer_15sec = time.time()
                        except Exception as e:
                            # Print the exception message
                            print(f"Error occurred while Updating Location: {e}")
            #Toggle Buzzer for Siren Sound
            if time.time() - time_3sec > 3:
                if buzzer_status == 1:
                    self.gpio.buzzer('off')
                else:
                    self.gpio.buzzer('on')
                time_3sec = time.time()
                buzzer_status *= -1
        
        ##Set Buzzer to OFF
        if buzzer_status == 1:
            self.gpio.buzzer('off')
        ##Send Lock Command to CarLock
        self.gpio.car_door_lock('lock')
        
        self.gpio.START_BUTTON_STATUS = False
        
        return

    #
    def successful_authentication(self,person_name,user_image=None):
        print('Sucess Authentication')
        #Send Successful Match notification in telegram with GPS Coordinates
        message = 'Successful Driver Authentication for '+person_name
        if user_image is None:
            self.telegram_bot.send_telegram_message(message)
        else:
            self.telegram_bot.send_telegram_photo(user_image,message,keyboard_required=False)
        gps_data = self.gps.fetch_GPS_Data()
        if 'Error' in gps_data:
            self.telegram_bot.send_telegram_message("Unable to Fetch GPS Coordinates")
        else:
            try:
                response = self.telegram_bot.send_telegram_live_location(gps_data[0],gps_data[1],0)
                if response['ok'] == True:
                    print('Location Sent Successfully')
                else:
                    print('Location Sent Failed')
            except Exception as e:
                # Print the exception message
                print(f"Error occurred while Sending Location: {e}")
        #Start Car Engine
        self.gpio.engine('on')
        ##Send Lock Command to CarLock
        self.gpio.car_door_lock('lock')
        
        self.configuration.config_file['FACE_AUTHENTICATION'] = True
        
        self.configuration.write_configuration_file()
        
        return

    #
    def failed_authentication(self,user_image):
        print('Failed Authentication')
        #Send Alert Notification with Captured Photo and GPS Coordinaties with Allow/Reject Option
        authentication = False
        caption = 'New User Detected'
        status,response = self.telegram_bot.send_telegram_photo(user_image,caption)
        print(status)
        # Extract message_id
        message_id = response['result']['message_id']
        
        # Extract chat id
        chat_id = response['result']['chat']['id']
        
        print("Message ID:", message_id)
        print("Chat ID:", chat_id)
        if status:
            input_recived,user_input,last_update_id=self.telegram_bot.check_user_input()
            if input_recived:
                self.configuration.config_file['LAST_UPDATE_ID'] = last_update_id
                self.configuration.write_configuration_file()
                if user_input == 'Allow':
                    self.successful_authentication('Guest')
                    authentication = True
                else:
                    self.start_AntiTheft_System()
            else:
                self.telegram_bot.edit_telegram_message_reply_markup(chat_id,message_id,'No response recieved in 1 min')
                self.telegram_bot.send_telegram_message("No response recieved in 1 min, Anti-Theft system started")
                self.start_AntiTheft_System()
        else:
            self.start_AntiTheft_System()
        
        return authentication
    
    def start_authentication(self,camera):
        print('Start Authentication')
        image_path = 'captured_frames/Face_cpatured.png'
        start = time.time()
        main_start = time.time()
        matched_name = self.face_class.capture_faces(camera,image_path,MAX_ATTEMPTS=3)
        print(matched_name)
        end = time.time()
        print('Image capturing in ',end-start)
        
        print('All Images Person identification in ',end-start)
        authentication = False
        if matched_name != 'Unknown':
            authentication = True
        print(matched_name,authentication)
        
        end = time.time()
        print('Complete Face recoginition in ',end-main_start)
        if authentication:
            self.successful_authentication(matched_name,image_path)
        else:
            authentication = self.failed_authentication(image_path)
            
        return authentication
    
    def live_bot_monitoring(self,camera,timeout_period=1):
        ret, frame = camera.read()
        updates = self.telegram_bot.get_telegram_update(timeout_period)
        if updates:
            print('Got Update')
            last_update_id = updates[-1]['update_id'] + 1
            self.telegram_bot.LAST_UPDATE_ID = last_update_id
            self.configuration.config_file['LAST_UPDATE_ID'] = last_update_id
            self.configuration.write_configuration_file()
            for update in updates:
                message = update.get('message')
                if not message:
                    continue
        
                text = message.get('text')
        
                if text:
                    text = text.lower()
                    if 'live picture' in text:
                        ret, frame = camera.read()
                        ret, frame = camera.read()
                        if not ret:
                            print("Error: Could not read frame")
                        else:
                            img_path = 'captured_frames/live_image'+str(int(time.time()))+'.jpg'
                            cv2.imwrite(img_path, frame)
                            print('Image captured ',img_path)
                            self.telegram_bot.send_telegram_photo(img_path,'Live Picture Captured',keyboard_required=False)
                            # os.remove(img_path)
                    elif 'location' in text:
                        gps_data = self.gps.fetch_GPS_Data()
                        response = self.telegram_bot.send_telegram_live_location(gps_data[0],gps_data[1],0)
                        if response['ok'] == True:
                            print('Location Sent Successfully')
                        else:
                            print('Location Sent Failed')
                    else:
                        self.telegram_bot.send_telegram_message("❗❗Unknown command. Please send 'Live picture' or 'Location'.")
        else:
            print('No update')
        
        return
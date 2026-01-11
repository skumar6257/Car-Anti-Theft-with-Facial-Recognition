# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:47:49 2024

@author: skuma
"""
##### NOTE: Uncomment all the commented section while running on Rasperry PI ####

# import serial
import pynmea2

class GPSClass:
    
    def __init__(self):
        #self.gps_ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)
        pass        
        
    #Fetching GPS Data
    def fetch_GPS_Data(self):
        ##To get input GPS data
        #Fetch input from GPS Module
        gps_data =[10.903228261463234, 76.89710647873268]
        # while True:
        #     line = self.gps_ser.readline().decode('utf-8', errors='ignore')
        #     if line[0:6] == "$GPRMC":
        #         newmsg=pynmea2.parse(line)
        #         lat=newmsg.latitude
        #         lng=newmsg.longitude
        #         gps_data = [lat,lng]
        #         break
        #     elif line == '':
        #         gps_data = 'Error in Fetching'
        #         break
        
        return gps_data
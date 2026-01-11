# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 09:59:03 2024

@author: skuma
"""
import json
from pathlib import Path

class ConfigurationClass:
    def __init__(self,filename):
        self.config_file = dict()
        self.file_name = filename
     
    ##Writing Configuration File
    def write_configuration_file(self):
        try:
            with open(self.file_name, 'w') as json_file:
                json.dump(self.config_file, json_file, indent=4)
        except Exception as e:
            # Print the exception message
            print(f"Error occurred while Writing Config File: {e}")
        return
        
    ##Loading Configuration File
    def load_configuration_file(self):

        file_path = Path(self.file_name)
        
        # Check if the file exists
        if file_path.exists():
            try:
                with open(self.file_name, 'r') as json_file:
                    self.config_file = json.load(json_file)
            except Exception as e:
                # Print the exception message
                print(f"Error occurred while Reading Config File: {e}")
                
        else:
            #Updating with Default value
            self.config_file['BOT_TOKEN'] = 'Fill your BOT Token'
            self.config_file['CHAT_ID'] = 'Fill your Chat ID'
            self.config_file['LAST_UPDATE_ID'] = None
            self.config_file['FACE_AUTHENTICATION'] = False
            
            self.write_configuration_file()
    
    

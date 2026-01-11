# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 11:17:18 2024

@author: skuma
"""
import requests
import json

class TelegramNotificationClass:
    
    def __init__(self,bot_token,chat_id,last_update_id):
        self.BOT_TOKEN = bot_token
        self.CHAT_ID = chat_id
        self.LAST_UPDATE_ID = last_update_id
        
        
    ##Send Telegram Message
    def send_telegram_message(self,message):
        
        try:
            url = "https://api.telegram.org/bot"+self.BOT_TOKEN+"/sendMessage"
            data = {'chat_id': self.CHAT_ID, 'text': message}
            response = requests.post(url, data=data)
            
            # Check for errors
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.json()}")
                
                return False
            else:
                print("Message sent successfully!")
                #print(f"Response: {response.json()}")
                
                return True
        
        except Exception as e:
            # Print the exception message
            print(f"Error occurred while Sending Message: {e}")
            return False
        
    ##Send Telegram Photo    
    def send_telegram_photo(self,image_filename,caption,keyboard_required=True):
        
        try:
            
            URL = "https://api.telegram.org/bot" + self.BOT_TOKEN + "/sendphoto"
            if keyboard_required:
                keyboard=json.dumps({ "inline_keyboard": 
                                     [ [ {"text": "✅ Allow", "callback_data": "Allow"}, 
                                        {"text": "❌ Reject", "callback_data": "Reject"}] ] })
            
            # Escape special characters for Markdown
            caption = caption.replace('_', '\\_')
            caption = caption.replace(':', '\\:')
    
            multipart_form_data = {
                'photo': (image_filename, open(image_filename, 'rb')),
                'action': (None, 'send'),
                'chat_id': (None, self.CHAT_ID),
                'caption': (None, caption),
            }
            
            if keyboard_required:
                keyboard=json.dumps({ "inline_keyboard": 
                                     [ [ {"text": "✅ Allow", "callback_data": "Allow"}, 
                                        {"text": "❌ Reject", "callback_data": "Reject"}] ] })
                multipart_form_data['parse_mode'] = (None, 'Markdown')
                multipart_form_data['reply_markup'] = (None, keyboard)
                
    
            response = requests.post(URL, files=multipart_form_data)
            
            # Check for errors
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.json()}")
                
                return False
            else:
                print("Photo sent successfully!")
                #print(f"Response: {response.json()}")
                if keyboard_required:
                    return True,response.json()
                else:
                    return True
        
        except Exception as e:
            # Print the exception message
            print(f"Error occurred while Sending Photo: {e}")
            return False
        
    def get_telegram_update(self,timeout):
        try:
        
            URL = "https://api.telegram.org/bot" + self.BOT_TOKEN + "/getUpdates"
            params = {'timeout': timeout, 'offset': self.LAST_UPDATE_ID}
            response = requests.get(URL, params=params)
            try:
                updates = response.json()['result']
            except:
                updates = []
        except Exception as e:
            # Print the exception message
            print(f"Error occurred while Getting Update: {e}")
            return False
        return updates

    def edit_telegram_message_reply_markup(self,chat_id,message_id,response_text):
        try:
        
            url = "https://api.telegram.org/bot"+ self.BOT_TOKEN + "/editMessageReplyMarkup"
            requests.post(url, data={
                'chat_id': chat_id,
                'message_id': message_id,
                'text': response_text,
                'reply_markup': json.dumps({'reply_keyboard': []})  # Remove the buttons
            })
            
        except Exception as e:
            # Print the exception message
            print(f"Error occurred while Edit Telegram Message Reply: {e}")

    ##Check User Input    
    def check_user_input(self):
        
        try:
            
            updates = self.get_telegram_update(60)
            user_input = None
            
            if updates:
                for update in updates:
                    if 'callback_query' in update:
                        callback_query_id = update['callback_query']['id']
                        data = update['callback_query']['data']
                        user_input = data
                        chat_id = update['callback_query']['message']['chat']['id']
                        message_id = update['callback_query']['message']['message_id']
    
                        if data == 'Allow':
                            response_text = 'You accepted image.'
                            # Perform the accept action here
                        elif data == 'Reject':
                            response_text = 'You rejected image.'
                            # Perform the reject action here
    
                        url = "https://api.telegram.org/bot"+ self.BOT_TOKEN +"/answerCallbackQuery"
                        requests.post(url, data={'callback_query_id': callback_query_id, 'text': response_text})
    
                        self.edit_telegram_message_reply_markup(chat_id,message_id,response_text)
                        
                        self.send_telegram_message(response_text)
                        
                    last_update_id = update['update_id'] + 1
                    self.LAST_UPDATE_ID = last_update_id
                    print(last_update_id)
                
            return (len(updates)!=0),user_input,self.LAST_UPDATE_ID
        
        except Exception as e:
            # Print the exception message
            print(f"Error occurred while Checking Input: {e}")
            return False,None,self.LAST_UPDATE_ID
    
    
    ##Send Live location
    def send_telegram_live_location(self,latitude,longitude,live_period):
        
        try:
        
            URL = "https://api.telegram.org/bot"+self.BOT_TOKEN+"/sendLocation"
            data = {
                'chat_id': self.CHAT_ID,
                'latitude': latitude,
                'longitude': longitude,
                'live_period': live_period
            }
            
            response = requests.post(URL, data=data)
            
            return response.json()
        
        except Exception as e:
            # Print the exception message
            print(f"Error occurred while Sending Live Location: {e}")
            return None
    
    ##Update Live Location
    def update_telegram_live_location(self,latitude,longitude,message_id):
        
        try:
        
            URL = "https://api.telegram.org/bot"+self.BOT_TOKEN+"/editMessageLiveLocation"
            data = {
                'chat_id': self.CHAT_ID,
                'latitude': latitude,
                'longitude': longitude,
                'message_id': message_id
            }
            response = requests.post(URL, data=data)
            
            return response.json()
        
        except Exception as e:
            # Print the exception message
            print(f"Error occurred while Updating Live Location: {e}")
            return None
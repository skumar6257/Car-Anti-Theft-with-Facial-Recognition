# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 10:25:17 2024

@author: skuma
"""

import os
import torch
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet, l2_norm
import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import glob
import torch.nn.functional as F

class FacialRecoginitionClass:
     
    def __init__(self,encoding_filename,image_directory):
        self.detect_model = None
        self.known_encodings = []
        self.known_names = []
        self.encoding_file_name = encoding_filename
        self.image_directory = image_directory
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    ##Loading Model
    def load_face_model(self):
        self.detect_model = torch.load('Weights/MobileFace_Net_with_weights.pth', map_location=self.device)
        self.detect_model.to(self.device)
        print('MobileFaceNet face detection model generated')
        self.detect_model.eval()
    
    def transformation_from_points(self,points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    def Face_alignment(self,img,default_square = True,landmarks = []):
        # face alignment -- similarity transformation
        faces = []
        if len(landmarks) != 0:
            for i in range(landmarks.shape[0]):
                landmark = landmarks[i, :]
                landmark = landmark.reshape(2, 5).T
    
                if default_square:
    
                    coord5point =  [[38.29459953, 51.69630051],
                                    [73.53179932, 51.50139999],
                                    [56.02519989, 71.73660278],
                                    [41.54930115, 92.3655014 ],
                                    [70.72990036, 92.20410156]]
    
                    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in landmark]))
                    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in coord5point]))
                    M = self.transformation_from_points(pts1, pts2)
                    aligned_image = cv2.warpAffine(img, M[:2], (img.shape[1], img.shape[0]))
                    crop_img = aligned_image[0:112, 0:112]
                    faces.append(crop_img)
    
                else:
    
                    coord5point =  [[30.29459953, 51.69630051],
                                    [65.53179932, 51.50139999],
                                    [48.02519989, 71.73660278],
                                    [33.54930115, 92.3655014],
                                    [62.72990036, 92.20410156]]
    
                    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in landmark]))
                    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in coord5point]))
                    M = self.transformation_from_points(pts1, pts2)
                    aligned_image = cv2.warpAffine(img, M[:2], (img.shape[1], img.shape[0]))
                    crop_img = aligned_image[0:112, 0:96]
                    faces.append(crop_img)
    
        return faces
    
    def visalize(self,img_path, bboxes, landmarks, names, inference_time,inference_flag=False):
    
        img = cv2.imread(img_path)
        
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('utils/simkai.ttf', 40)
        if inference_flag:
            # Draw the inference time at the top-left corner of the frame
            inference_time_text = f"Inference Time: {inference_time} s"
            draw.text((10, 10), inference_time_text, fill=(0, 255, 255), font=font)
        outline = 'blue'
        for i, b in enumerate(bboxes):
            if names[i] == 'Unknown':
                outline = 'red'
            else:
                outline = 'green'
            draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline=outline, width=5)
            draw.text((int(b[0]), int(b[1]-50)), names[i], fill=(255,255,0), font=font)
            
        # Save the image with bounding boxes and names
        image.save(img_path, format='JPEG')
    #     print(f"Image saved to {img_path}")
    
        image_ = np.array(image)
  
        # Convert PIL Image back to OpenCV format (BGR)
        img_cv2 = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR)
        
        return img_cv2
    
    def detect_faces(self,img):
    # Detect faces
        bboxes, landmarks = create_mtcnn_net(img, 20, self.device,
                                         p_model_path='Weights/PNet_with_weights.pth',
                                         r_model_path='Weights/RNet_with_weights.pth',
                                         o_model_path='Weights/ONet_with_weights.pth')
        
        return bboxes, landmarks
    
    def generate_facial_embeddings(self,aligned_img):
        
        test_transform = trans.Compose([
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
        with torch.no_grad():
            mirror = cv2.flip(aligned_img, 1)
            emb = self.detect_model(test_transform(aligned_img).to(self.device).unsqueeze(0))
            emb_mirror = self.detect_model(test_transform(mirror).to(self.device).unsqueeze(0))
        
        return l2_norm(emb + emb_mirror)
        
    ##Loading Encoding Files
    def load_facial_encodings(self):
        print('Load Encodings')
        
        
        file_path = Path(self.encoding_file_name)
            
        # Check if the file exists
        if file_path.exists():
            try:
                # Load the encodings from the .pth file
                data = torch.load(self.encoding_file_name)

                # Extract the embeddings and names
                self.known_encodings = data['embeddings']
                self.known_names = data['names']
            except Exception as e:
                # Print the exception message
                print(f"Error occurred while Reading Encoding File: {e}")
        
        else:
            #Create Encoding File
            #Creating Facial Encoding Of Known Users From Database
            images=[f for f in os.listdir(self.image_directory) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            names = []
            encodings = []
            for image_filename in images:
                name = image_filename.split('.')[0]
                names.append(name)
                
                #Open Image
                image_path = os.path.join(self.image_directory,image_filename)
                image = cv2.imread(image_path)
                
                start = time.time()
                # Detect faces
                bboxes, landmarks = self.detect_faces(image)
                
                #Face Alignment
                aligned_faces = self.Face_alignment(image, default_square=True, landmarks=landmarks)
                
                #Considering Only one face as creating Known user Database
                aligned_img = aligned_faces[0]   
                
                #Generate Facial Embeddings
                encoding=self.generate_facial_embeddings(aligned_img)
                encodings.append(encoding)
                
                end = time.time()
                print(name,'Face detected in',name,'Time Taken',(end - start),'s') # time in seconds
        
            try:
                self.known_encodings = torch.cat(encodings)
                self.known_names = np.array(names)
                encoding_data ={'names':self.known_names,'embeddings':self.known_encodings}
                torch.save(encoding_data,self.encoding_file_name)
                print(f"Encodings saved to {self.encoding_file_name}")
            except Exception as e:
                # Print the exception message
                print(f"Error occurred while Writing Encoding File: {e}")
                
    def L2_distance_below_threshold(self,source_embeddings,known_embeddings,threshold):
        
        diff = source_embeddings.unsqueeze(-1) - known_embeddings.transpose(1, 0).unsqueeze(0) # i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
        dist = torch.sum(torch.pow(diff, 2), dim=1) # number of detected faces x numer of target faces
        minimum, min_idx = torch.min(dist, dim=1) # min and idx for each row
        print(dist)
        for face_min in minimum:
            if face_min > threshold:
                min_idx[minimum > threshold+0.1] = -1  # if no match, set idx to -1
            else:
                min_idx[minimum > threshold] = -1  # if no match, set idx to -1
        score = minimum
        results = min_idx
        
        return results

    def cosine_similarity_below_threshold(self,source_embeddings, known_embeddings, threshold=0.5):
        if source_embeddings.ndim == 1:
            source_embeddings = source_embeddings.unsqueeze(0)  # Add batch dimension
    
        # Normalize the embeddings to unit vectors
        source_embeddings = F.normalize(source_embeddings, p=2, dim=1)  # Normalize source embeddings
        known_embeddings = F.normalize(known_embeddings, p=2, dim=1)    # Normalize known embeddings
    
        # Compute cosine similarity: dot product of normalized vectors
        similarity = torch.matmul(source_embeddings, known_embeddings.T)  # Shape: (num_detected_faces, num_target_faces)
        print(similarity)
    
        # Find the maximum similarity for each detected face
        maximum, max_idx = torch.max(similarity, dim=1)  # Max similarity and index for each row

        for face_max in maximum:
            if face_max < threshold:
                max_idx[maximum < threshold-0.05] = -1  # if no match, set idx to -1
            else:
                max_idx[maximum < threshold] = -1  # if no match, set idx to -1
    
        score = maximum
        results = max_idx
    
        return results
                
                
    ##Check Matched Faces
    def check_facial_match(self,image_filename):
        print('Check facial match')
        matched = False
        matched_names_idx = []
        matched_names =[]
        color='red'
        
        #Open Image
        image = cv2.imread(image_filename)
        
        start = time.time()
        # Detect faces
        bboxes, landmarks = self.detect_faces(image)
        end_time = time.time()
    #     print('detect_faces in ',end_time - start)
        if len(bboxes) != 0:
            start = time.time()
            #Face Alignment
            aligned_faces = self.Face_alignment(image, default_square=True, landmarks=landmarks)
    
            embeddings = []
    
            #Generate Facial Embeddings
            for face in aligned_faces:
                encoding=self.generate_facial_embeddings(face)
                embeddings.append(encoding)
            embeddings = torch.cat(embeddings)
        
            #Identify Matched Face
            # matched_names_idx=self.L2_distance_below_threshold(embeddings,self.known_encodings,1.0)
            matched_names_idx=self.cosine_similarity_below_threshold(embeddings,self.known_encodings,0.5)
            print('Matched Name in Facial Match ',matched_names_idx)
    
            for idx in matched_names_idx:
                if idx == -1:
                    matched_names.append('Unknown')
                else:
                    print('Id:',idx,'Names:',self.known_names)
                    name = self.known_names[idx]
                    name = ' '.join(name.split('_')[:2])
                    matched_names.append(name)
        else:
            matched_names = ['Unknown']
        
        end = time.time()
        inference_time = round(end - start,4)
    #     print('Face Processing and Matching in',image_filename,'Time Taken',inference_time,'s') # time in seconds
        
            
        return bboxes,matched_names,inference_time
    
    #Check for Blur Free
    def is_blur_free(self,image, threshold=100.0):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var > threshold
    
    ##Capturing Live Image
    def capture_faces(self,camera,frame_path,MAX_ATTEMPTS=10):
        print('Capture faces')
        # Create a directory to save the captured frames
        if os.path.exists(frame_path):
            # Remove the file
            os.remove(frame_path)
            print(f"The file {frame_path} has been removed.")
            
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            
        unknown_count = 0
        matched_name = 'Unknown'
        blur_threshold = 100.0
        flag_count = 0
       
        if not camera.isOpened():
            raise Exception("Could not open video device")
        
        while True:
            # Capture frame-by-frame
            ret, frame = camera.read()
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
            # Perform face detection
            start = time.time()
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            end = time.time()
    
            if len(faces) != 0:
    
                if self.is_blur_free(frame,blur_threshold):
                    # Save the frame to the directory
                    cv2.imwrite(frame_path, frame)
                    bboxes,names,inference_time = self.check_facial_match(frame_path)
                    self.visalize(frame_path,bboxes,None,names,inference_time,inference_flag=False)
                    print(names)
                    for name in names:
                        if name == 'Unknown':
                            unknown_count += 1//len(names)
                            print(unknown_count)
                        else:
                            matched_name = name
                            break
                            
                    if matched_name != 'Unknown':
                        break
                else:
                    flag_count += 1
                    if flag_count%10 == 0:
                        print(blur_threshold)
                        blur_threshold -= 20
                    
            # Display the resulting frame
            cv2.imshow('Webcam - Face Detection', frame)
    
            # Break the loop if 'q' is pressed or face is detected
            if cv2.waitKey(1) & unknown_count == MAX_ATTEMPTS:
                break
        
        # Release the capture and close the window
        print(unknown_count)
        cv2.destroyAllWindows()
    
        return matched_name
    
    def live_face_check(self,camera):
        #Live Name and Infernce Time Check
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        frame_path='captured_frames/Face_cpatured.png'       
        
        while True:
            # Capture frame-by-frame
            ret, frame = camera.read()
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # Perform face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
            if len(faces) != 0:
        
                if self.is_blur_free(frame,100.0):
                    # Save the frame to the directory
                    cv2.imwrite(frame_path, frame)
                    bboxes,names,inference_time = self.check_facial_match(self.detect_model,frame_path)
                    frame = self.visalize(frame_path,bboxes,None,names,inference_time,inference_flag=False)
        
        
            # Display the resulting frame
            cv2.imshow('Webcam - Face Detection', frame)
        
            # Break the loop if 'q' is pressed or face is detected
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the capture and close the window
        cv2.destroyAllWindows()
        

        return

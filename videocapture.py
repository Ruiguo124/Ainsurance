import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import time
import os

emotion = {0:'Angry',1:'Disgust', 2:'Fear', 3 :'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
model = tf.keras.models.load_model('fer.h5')
max_index=0
IMG_SIZE = 48
depressionRate = 0
depressionRateNeutral = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#mixer.init()
#mixer.music.load('darkness.mp3')

cap = cv2.VideoCapture(0)
baseMoney = 250
frameCount = 0
frameSad = 0
frameNeutral = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frameCount = frameCount + 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    faces = np.asarray(faces)
    print(faces)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)] 
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
        detected_face = cv2.resize(detected_face, (48, 48)) 
        #detected_face[0] /= 255
        # img_pixels = np.array(detected_face)
        # print(img_pixels)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
		
        img_pixels /= 255 
        
        predictions = model.predict(img_pixels) 
		 
        max_index = np.argmax(predictions[0])
        
        
		
		
        cv2.putText(frame, emotion[int(max_index)], (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
    if emotion[int(max_index)]== 'Sad' or emotion[int(max_index)] == 'Fear' or emotion[int(max_index)]== 'Angry':
        frameSad = frameSad+1

    if emotion[int(max_index)]== 'Neutral':
        frameNeutral = frameNeutral+1
        
    if baseMoney > 1000:
        os.system('mpg123 darkness.mp3')
        
    
    if frameCount > 0:
        depressionRateSad = (frameSad/frameCount) * 100
    if frameCount > 0:
        depressionRateNeutral = (frameNeutral/frameCount)*100
        print(depressionRateNeutral)
    if depressionRate > 25:
        #mixer.music.play()
        baseMoney  = baseMoney + (baseMoney * 0.05)/60
            
    
    if depressionRateNeutral > 30:
        baseMoney  = baseMoney + (baseMoney * 0.0001)/60
            
    text = "Insurance price : " + str(baseMoney)
    cv2.putText(frame, text, (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,10), 2)
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
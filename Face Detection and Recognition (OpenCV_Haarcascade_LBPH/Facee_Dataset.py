import cv2
import PIL
import numpy
import os


# Giving access to webcam on my pc
camera = cv2.VideoCapture(0)
camera.set(3, 640) #width in pxl
camera.set(4,480) # height in pxl

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # this will able opencv to detect the face
face_id = input('\n Enter userID and press enter')
print("\n Initializing face capture.")

count = 0
while (True):
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting the images to gray scale 
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # writing the images to the dataset 
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h ), (255,0,0), 2) # prefix
        count += 1
        cv2.imwrite('Data/User.' + str(face_id) + '.' + str(count) + '.jpg', gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    #initializing the waiting time for the user to capture the images
    waiting_time = cv2.waitKey(100) & 0xff
    if waiting_time == 10:
        break
    elif count >= 30:
        break

print("\n Exiting program")
camera.release()
cv2.destroyAllWindows
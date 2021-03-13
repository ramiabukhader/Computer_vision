import cv2
import numpy as np
from PIL import Image
import os

pathh = 'Data'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImages_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_sampels = []
    ids = []

    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for(x,y,w,h) in faces:
            face_sampels.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    
    return face_sampels, ids

print("\n Training faces...")
faces, ids = getImages_and_labels(pathh)
recognizer.train(faces, np.array(ids))

recognizer.write('Trainer/trainer,yml')

print("\n {0} faces trained.".format(len(np.unique(ids))))
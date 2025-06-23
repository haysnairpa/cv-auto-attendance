import os
from os import listdir
from PIL import Image
from numpy import asarray, expand_dims
import numpy as np
import pickle
import cv2
from keras_facenet import FaceNet

# Initialize Haar Cascade and FaceNet model
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

folder = 'pics/'  # Folder containing images
database = {}

# Iterate through each file in the folder
for filename in listdir(folder):
    path = folder + filename
    gbr1 = cv2.imread(path)  # Read the image

    # Detect faces in the image
    faces = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

    # If a face is detected, get its coordinates; otherwise, set default coordinates
    if len(faces) > 0:
        x1, y1, width, height = faces[0]
    else:
        x1, y1, width, height = 1, 1, 10, 10

    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    # Convert BGR image to RGB
    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    gbr = Image.fromarray(gbr)  # Convert from OpenCV to PIL
    gbr_array = asarray(gbr)  # Convert PIL image to NumPy array

    # Extract the face region from the image
    face = gbr_array[y1:y2, x1:x2]

    face = Image.fromarray(face)
    face = face.resize((160, 160))  # Resize the face image to 160x160 pixels
    face = asarray(face)  # Convert PIL image to NumPy array

    face = expand_dims(face, axis=0)  # Add batch dimension

    embeds = MyFaceNet.embeddings(face)  # Generate embeddings using FaceNet

    # Use the part before the first underscore as the key
    base_name = os.path.splitext(filename)[0].split('_')[0]
    if base_name not in database:
        database[base_name] = []
    database[base_name].append(embeds)

    '''
    database[os.path.splitext(filename)[0]] = embeds'''

# Save the database to a pkl file
with open("embeddings/data.pkl", "wb") as myfile:
    pickle.dump(database, myfile)

print("Face embeddings saved successfully to data.pkl!")

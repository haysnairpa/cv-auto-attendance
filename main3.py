import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import datetime
import os
import pickle
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Check if models exist before loading
if not os.path.exists('model/gender_detection.h5'):
    raise FileNotFoundError("Model gender_detection20.h5 not found in the 'model' folder!")

if not os.path.exists('model/age_model_efficientnet.h5'):
    raise FileNotFoundError("Model age_model_balanced_v2.h5 not found in the 'model' folder!")

# Load trained models
gender_model = load_model('model/gender_detection20.h5')
age_model = load_model('model/age_model_efficientnet.h5', custom_objects={'mse': MeanSquaredError})

# Check the expected input shape for the age model
expected_age_input_shape = age_model.input_shape[1:3]  # (height, width)
print(f"Expected Age Model Input Shape: {expected_age_input_shape}")

# Initialize FaceNet and Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
MyFaceNet = FaceNet()

# Load face embedding database
if not os.path.exists("embeddings/data.pkl"):
    raise FileNotFoundError("Face database 'embeddings/data.pkl' not found!")

with open("embeddings/data.pkl", "rb") as myfile:
    database = pickle.load(myfile)

# Create attendance file
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"attendance_{current_time}.xlsx"
detected_faces = {}

# Initialize camera
cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

processed_faces = set()  # Track processed faces

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(frame_rgb)

    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            x, y = max(0, x - 20), max(0, y - 20)
            w, h = min(iw - x, w + 40), min(ih - y, h + 40)
            
            face_key = (x, y, w, h)  # Unique identifier for the face
            
            if face_key in processed_faces:
                continue  # Skip already processed faces
            
            # Extract face from the frame
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue  # Skip if the face is invalid
            
            # **Face Recognition**
            face_resized = cv2.resize(face_img, (160, 160))
            face_array = np.expand_dims(face_resized, axis=0)
            signature = MyFaceNet.embeddings(face_array)
            identity = "Unknown"
            min_dist = 100
            
            for key, value in database.items():
                distances = [np.linalg.norm(embedding - signature) for embedding in value]
                avg_dist = np.mean(distances)
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    identity = key
            
            # **Gender Prediction**
            gender = "Unknown"  # Default gender if prediction fails
            try:
                face_resized_for_gender = cv2.resize(face_img, (64, 64))
                face_array_for_gender = np.expand_dims(face_resized_for_gender, axis=0) / 255.0
                predicted_gender = gender_model.predict(face_array_for_gender)
                gender = 'Male' if predicted_gender[0][0] < 0.4 else 'Female'
            except Exception as e:
                print(f"❌ Error during gender prediction: {e}")

            # **Age Prediction**
            predicted_age = "Unknown"  # Default age if prediction fails
            try:
                # Resize according to the age model's input shape
                face_resized_for_age = cv2.resize(face_img, expected_age_input_shape)
                face_array_for_age = np.expand_dims(face_resized_for_age, axis=0) / 255.0
                
                # Check if the model requires flattening
                if len(age_model.input_shape) == 2:  # If the model input shape is (None, 9216)
                    face_array_for_age = face_array_for_age.reshape(1, -1)  # Reshape to 1D vector

                predicted_age = age_model.predict(face_array_for_age)
                predicted_age = int(np.clip(predicted_age[0][0], 1, 100))  # Correct age range
            except Exception as e:
                print(f"❌ Error during age prediction: {e}")

            print(f"Predicted Age: {predicted_age}")  # Debugging output
            
            # Save attendance data
            if identity != "Unknown" and identity not in detected_faces:
                detected_faces[identity] = (datetime.datetime.now().strftime("%H:%M:%S"), predicted_age)
                df = pd.DataFrame([(k, v[0], v[1]) for k, v in detected_faces.items()], columns=['Name', 'Time', 'Age'])
                df.to_excel(filename, index=False)
                print(f"✅ {identity} ({gender}, {predicted_age} years) recorded at {detected_faces[identity][0]}")
            
            processed_faces.add(face_key)  # Mark the face as processed
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Ensure gender & age do not cause errors before use
            gender_age_text = f"{gender}, Age: {predicted_age}" if gender != "Unknown" and predicted_age != "Unknown" else "Unknown Data"
            
            # Display text (identity, gender, age)
            text_y = y - 10 if y - 10 > 20 else y + h + 20
            cv2.putText(frame, f"{identity} ({gender_age_text})", (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv2.imshow("Face Verification", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
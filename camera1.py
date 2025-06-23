import cv2
import os

# Initialize Haar Cascade model
HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def capture_and_save_images(name):
    video = cv2.VideoCapture(0)
    cv2.namedWindow("Capture Face Data")

    image_count = 0
    total_images = 5

    # Ensure the pics folder exists
    if not os.path.exists('pics'):
        os.makedirs('pics')

    while image_count < total_images:
        ret, frame = video.read() # Read a frame from the camera
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = HaarCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.shape[0] > 0 and face_img.shape[1] > 0:  # Ensure the face image is not empty
                image_count += 1
                # Save the face image with the specified naming format
                image_path = f'pics/{name}_{str(image_count).zfill(2)}.png'
                cv2.imwrite(image_path, face_img)
                print(f"Saved {image_path}")
                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capture Face Data", frame) # Display the frame with rectangles
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if image_count >= total_images:
            print("Captured all images.")
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter your name: ")
    capture_and_save_images(name)

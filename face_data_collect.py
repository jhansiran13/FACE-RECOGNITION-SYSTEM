import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = "./data/"
file_name = input("Enter the name of the person: ")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Process frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    if len(faces) == 0:
        continue

    faces = sorted(faces, key=lambda f: f[2] * f[3])
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        offset = 10
        h_frame, w_frame = frame.shape[:2]
        x_start = max(x - offset, 0)
        y_start = max(y - offset, 0)
        x_end = min(x + w + offset, w_frame)
        y_end = min(y + h + offset, h_frame)
        face_section = frame[y_start:y_end, x_start:x_end]
         # Check if the cropped section is valid
        if face_section.size == 0:
            print("Invalid face section, skipping...")
            continue

        # Resize face section to 100x100
        face_section = cv2.resize(face_section, (100, 100))
        # Save every 10th frame
        skip += 1
        if skip % 10 == 0:
            face_data.append(face_section)
            print(f"Captured {len(face_data)} frames.")

    cv2.imshow("Frame", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Save data
face_data = np.asarray(face_data).reshape((len(face_data), -1))
np.save(dataset_path + file_name + '.npy', face_data)
print("Data saved successfully.")

cap.release()
cv2.destroyAllWindows()

# AI Generated
#   Mistral Le Chat
"""
I want to write a script to read from my webcam, store the images as an np array for 5 seconds or 
so, and then re-write this image to a file which can be reloaded later. 
Can you help me outline this?
"""
import cv2
import numpy as np
import time

# Step 1: Initialize the webcam
cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Step 2: Capture images for 5 seconds
duration = 15  # seconds
start_time = time.time()
images = []

while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    images.append(frame)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 3: Convert the list of images to a NumPy array
images_array = np.array(images)

# Step 4: Save the NumPy array to a file
np.save('captured_images.npy', images_array)
print("Images saved to 'captured_images.npy'")

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()

# Step 5: Load the NumPy array from the file
loaded_images_array = np.load('captured_images.npy')
print("Images loaded from 'captured_images.npy'")

# Optional: Display the loaded images
for img in loaded_images_array:
    cv2.imshow('Loaded Image', img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

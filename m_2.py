import cv2
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load pre-trained Haarcascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained mask detection model
mask_model = load_model('mask_detector.model')

# Function to detect faces and predict if they are wearing a mask using MobileNetV2
def detect_mask(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Unable to read image file '{image_path}'.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        print(f"Face found at: X:{x}, Y:{y}, Width:{w}, Height:{h}")
        # Draw rectangles around each face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define region of interest (ROI) for face
        roi_color = img[y:y + h, x:x + w]

        # Preprocess the face region for mask detection with MobileNetV2 requirements
        face = cv2.resize(roi_color, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Predict mask/no mask
        (mask, withoutMask) = mask_model.predict(face)[0]
        print(f"Mask confidence: {mask}, WithoutMask confidence: {withoutMask}")

        # Determine the class label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Add label above the rectangle
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the image with detections
    cv2.imshow('Face Mask Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'C:\\Users\DP\\Documents\\DP\\Study\\learn\\Mask\\images\\MaskedorUnmasked\\Masked\\Women-wearing-facemasks-while-walking-outdoors-Milan-Italy-February-2020-coronavirus-COVID-19_jpg.rf.53e7a253c43d1af48e38ddddd315e982.jpg'
detect_mask(image_path)
image_path = 'C:\\Users\DP\\Documents\\DP\\Study\\learn\\Mask\\images\\MaskedorUnmasked\\Unmasked\\1000_F_285141218_O94xJsBP9fKOohaH3IfpVa57O8ThyHGD_jpg.rf.bd95484a1862741a19874a0f59088089.jpg'
detect_mask(image_path)
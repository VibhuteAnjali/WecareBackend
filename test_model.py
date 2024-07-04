
import os
import numpy as np
import cv2
from skimage import feature
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to quantify an image using Histogram of Oriented Gradients (HOG)
def quantify_image(image):
    features = feature.hog(image, orientations=9, pixels_per_cell=(10, 10),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    return features

# Function to load and quantify images
def load_and_quantify_images(folder_path):
    images = []
    labels = []
    for img_path in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, img_path), cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        img = cv2.resize(img, (200, 200))  # Resize image to (200, 200)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # Apply thresholding
        features = quantify_image(img)  # Extract HOG features
        images.append(features)
        labels.append(folder_path.split('-')[-1])  # Extracting label from folder name
    return images, labels

# Function to predict if a person is healthy or has Parkinson's disease
def predict_health_status(image):
    # Load trained model
    model = RandomForestClassifier(n_estimators=100)
    
    # Train the model with available data
    spiral_train_healthy = "healthy"
    spiral_train_park = "parkinson"
    trainX_healthy, trainY_healthy = load_and_quantify_images(spiral_train_healthy)
    trainX_park, trainY_park = load_and_quantify_images(spiral_train_park)
    trainX = np.array(trainX_healthy + trainX_park)
    trainY = np.array(trainY_healthy + trainY_park)
    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    model.fit(trainX, trainY)

    # Make prediction
    new_img = cv2.resize(image, (200, 200))  # Resize image to (200, 200)
    new_img = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # Apply thresholding
    new_features = quantify_image(new_img)  # Extract HOG features
    prediction = model.predict([new_features])[0]
    predicted_label = le.inverse_transform([prediction])[0]

    # Return prediction result
    if predicted_label == 'healthy':
        return 00
    else:
        return 80

# Corrected image path
# image_path = 'test-parkinson\V06PE01.png'  # Replace 'test-parkinson/V01PE01.png' with the actual path to your image file
# img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # Call the function to predict the health status
# result = predict_health_status(img)

# # Print the result
# print(result)


__all__ = ["load_and_quantify_images", "predict_health_status"]
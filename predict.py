import sys
import cv2
import joblib
import numpy as np
import os

def extract_hog_features(image, resize_shape=(64, 128)):
    resized_img = cv2.resize(image, resize_shape)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    
    hog_features = hog.compute(gray_img)
    return hog_features.flatten()

def predict_denomination(image_path, model_path="currency_svm_model.pkl"):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run 'python train.py' first to generate the model.")
        return

    try:
        data = joblib.load(model_path)
        svm_model = data['model']
        classes = data['classes']
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return
        
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
        
    try:
        features = extract_hog_features(image)
        features = features.reshape(1, -1) # Reshape for a single sample prediction
        
        prediction = svm_model.predict(features)[0]
        
        print(f"Image: {image_path}")
        print(f"Predicted Denomination: {prediction}")
        
        # Display the image with the prediction
        display_img = cv2.resize(image, (600, int(600 * image.shape[0] / image.shape[1])))
        cv2.putText(display_img, f"Predicted: {prediction}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Prediction Result", display_img)
        print("Press any key on the image window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    predict_denomination(img_path)

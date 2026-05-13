import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def extract_hog_features(image, resize_shape=(64, 128)):
    """
    Resize image, convert to grayscale, and extract HOG features.
    """
    # Resize image. Note: resize takes (width, height) in OpenCV.
    resized_img = cv2.resize(image, resize_shape)
    # Convert to grayscale
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    
    # Initialize HOG descriptor
    # Standard HOG parameters for a 64x128 image
    hog = cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    
    # Extract HOG features
    hog_features = hog.compute(gray_img)
    return hog_features.flatten()

def load_data(dataset_path):
    X = []
    y = []
    classes = []
    
    print(f"Loading data from {dataset_path}...")
    
    # Identify classes based on folder names
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            classes.append(folder_name)
            
    classes.sort()
    print(f"Found classes: {classes}")
    
    for class_name in classes:
        folder_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"Extracting features for class '{class_name}' ({len(image_files)} images)...")
        
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(folder_path, img_file)
            image = cv2.imread(img_path)
            
            if image is not None:
                try:
                    # Width=64, height=128
                    features = extract_hog_features(image, resize_shape=(64, 128))
                    X.append(features)
                    y.append(class_name)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            else:
                print(f"Failed to read {img_path}")
                
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(image_files)}")
                
    return np.array(X), np.array(y), classes

if __name__ == "__main__":
    dataset_dir = "dataset"
    model_filename = "currency_svm_model.pkl"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        print("Please place your images in the 'dataset' folder, organized by denomination.")
        exit(1)
        
    X, y, classes = load_data(dataset_dir)
    
    if len(X) == 0:
        print("No images found or processed successfully. Exiting.")
        exit(1)
        
    print(f"\nTotal samples: {len(X)}")
    print(f"Feature vector length: {len(X[0])}")
    
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training SVM model... (This may take a while depending on dataset size)")
    # Using a linear kernel as a starting point. It's often effective for HOG features.
    svm_model = SVC(kernel='linear', C=1.0, random_state=42) 
    svm_model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = svm_model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc * 100:.2f}%")
    
    print(f"\nSaving model to {model_filename}...")
    joblib.dump({'model': svm_model, 'classes': classes}, model_filename)
    print("Done!")

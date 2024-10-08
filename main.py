import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from mtcnn import MTCNN
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenet_model = hub.KerasLayer(model_url, input_shape=(224, 224, 3))

detector = MTCNN()

def extract_face(image, required_size=(224, 224)):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None
    
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    face = image[y1:y2, x1:x2]
    
    face_image = Image.fromarray(face)
    face_image = face_image.resize(required_size)
    face_array = np.asarray(face_image)
    
    return face_array

def preprocess_input(image):
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def get_embedding(face_pixels):
    face_pixels = preprocess_input(face_pixels)
    embedding = mobilenet_model(face_pixels)
    normalized_embedding = normalize(embedding)[0]
    return normalized_embedding

def compute_dataset_embeddings(directory):
    embeddings = []
    filenames = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        print(f"Processing file: {path}")
        try:
            image = cv2.imread(path)
            if image is None:
                print(f"Warning: Unable to read {path}. Skipping.")
                continue
            
            face = extract_face(image)
            if face is not None:
                embedding = get_embedding(face)
                embeddings.append(embedding)
                filenames.append(filename)
            else:
                print(f"No face detected in {filename}. Skipping.")
        except Exception as e:
            print(f"Error processing {filename}: {e}. Skipping.")
    
    return np.array(embeddings), filenames

def recognize_face_in_dataset(query_image_path, dataset_directory):
    dataset_embeddings, dataset_filenames = compute_dataset_embeddings(dataset_directory)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(dataset_embeddings)
    
    query_image = cv2.imread(query_image_path)
    query_face = extract_face(query_image)

    if query_face is not None:
        query_embedding = get_embedding(query_face)
        distances, indices = nbrs.kneighbors([query_embedding])
        print(f"Closest match: {dataset_filenames[indices[0][0]]} with distance {distances[0][0]}")
        
        threshold = 10.5  # Adjusting this I can specify the accuracy required to make the match or not
        if distances[0][0] < threshold:
            return f"Match found: {dataset_filenames[indices[0][0]]}"
        else:
            return "No match found."
    else:
        return "No face detected in the query image."

dataset_directory = 'C:\\Users\\Max\\Desktop\\Photolist' 
query_image_path ='C:\\Users\\Max\\Desktop\\5d614154-22f1-4f54-8f57-cc24d0770179.jpg'  
result = recognize_face_in_dataset(query_image_path, dataset_directory)
print(result)


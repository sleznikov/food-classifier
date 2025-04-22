import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

# Constants
IMG_SIZE = 224

def load_saved_model():
    # Load the trained model
    model = tf.keras.models.load_model('models/final_model.h5')  
    
    # Load class indices
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    return model, class_indices

def predict_image(model, image_path, class_indices):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    # Get class name from index
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    
    return predicted_class_name, confidence


if __name__ == "__main__":
    # Load the saved model
    model, class_indices = load_saved_model()
    
    # Make predictions on new images
    test_image_path = 'path_to_your_test_image.jpg'
    predicted_class, confidence = predict_image(model, test_image_path, class_indices)
    
    print(f"Predicted food: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

    test_images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    for img_path in test_images:
        predicted_class, confidence = predict_image(model, img_path, class_indices)
        print(f"\nImage: {img_path}")
        print(f"Predicted food: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")

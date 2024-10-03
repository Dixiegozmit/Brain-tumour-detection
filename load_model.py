import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('brain_tumor_model.h5')

# Function to preprocess the image
def preprocess_image(img_path):
    # Load the image with the target size
    img = image.load_img(img_path, target_size=(150, 150))  # Adjust size to 150x150
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Reshape it to fit the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image (if you did this during training)
    img_array /= 255.0  # Scale pixel values to [0, 1]
    return img_array


def predict(img_path):
    
    processed_img = preprocess_image(img_path)
    
    prediction = model.predict(processed_img)

    print("Prediction probabilities:", prediction)
    
    class_labels = ['No Tumor', 'Tumor']  
    predicted_class = class_labels[int(prediction[0][0] > 0.5)]  
    print(f"Predicted class: {predicted_class}")

# Example usage
if __name__ == "__main__":
    # Provide the path to an image you want to test
    img_path ='C:\\Users\\Acer\\OneDrive\\Desktop\\images\\5b39f3d8b4d1c924316deceaeeabe1_big_gallery.jpeg' 
    predict(img_path)
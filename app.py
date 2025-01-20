import streamlit as st
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from keras.models import load_model
from torch import nn

# Set parameters
HEIGHT = 32
WIDTH = 55
N_CHANNELS = 3  # RGB
BATCH_SIZE = 32
EPOCHS = 25
CATEGORIES = ['Buffalo', 'Elephant', 'Rhino', 'Zebra']
path = 'animal_dataset'

# Load Keras model
keras_model = load_model('animal_classification_model.h5')

# Define CNN model
class AnimalCNN(nn.Module):
    def __init__(self):
        super(AnimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 13, 128)  # Adjusted for your input size
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, len(CATEGORIES))

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch_model = AnimalCNN().to(device)
torch_model.load_state_dict(torch.load('animal_cnn_model.pth', map_location=device))
torch_model.eval()

def preprocess_image(image):
    """Preprocess the uploaded image for both models."""
    image_resized = cv2.resize(image, (WIDTH, HEIGHT))
    image_scaled = image_resized.astype('float32') / 255.0
    keras_input = image_scaled.reshape(1, HEIGHT, WIDTH, N_CHANNELS)

    torch_input = torch.tensor(image_scaled.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return keras_input, torch_input

def predict_keras(image):
    """Predict using Keras model."""
    pred = keras_model.predict(image)
    predictions = np.argmax(pred, axis=1)
    return CATEGORIES[predictions[0]]

def predict_torch(image):
    """Predict using PyTorch model."""
    with torch.no_grad():
        output = torch_model(image)
        _, pred = torch.max(output, 1)
    return CATEGORIES[pred.item()]

# Streamlit App
st.title("Animal Classification App")
st.write("Upload an image to classify using both Keras and PyTorch models.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    keras_input, torch_input = preprocess_image(image_rgb)

    # Make predictions
    keras_result = predict_keras(keras_input)
    torch_result = predict_torch(torch_input)

    # Display results
    st.write("### Predictions")
    st.write(f"**Keras Model Prediction:** {keras_result}")
    st.write(f"**PyTorch Model Prediction:** {torch_result}")

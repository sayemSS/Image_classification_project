# train_model.py
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import gradio as gr

# Config
MODEL_PATH = "human_animal_model.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# Build simple CNN model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # binary output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_and_save_model():
    print("Training model... Make sure dataset/train & dataset/test exist!")
    
    # Data Augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescale for test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        "dataset/train", 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='binary'
    )
    
    test_gen = test_datagen.flow_from_directory(
        "dataset/test", 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='binary'
    )

    model = build_model()
    # Increase epochs to 10 for better training
    model.fit(train_gen, epochs=10, validation_data=test_gen)
    model.save(MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
    return model

# Load model if exists, else train a new one
if os.path.exists(MODEL_PATH):
    print("Model found. Loading existing model.")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("No model found. Starting new training.")
    model = train_and_save_model()

# Prediction function for Gradio
def predict_image(img):
    if img is None:
        return {"Human": 0.0, "Animal": 0.0}
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img)/255.0, 0)
    human_prob = float(model.predict(arr)[0][0])
    return {"Human": round(human_prob, 4), "Animal": round(1-human_prob, 4)}

# Gradio UI
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Human vs Animal Classifier",
    description="Upload an image and see if it's Human or Animal."
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", share=False)
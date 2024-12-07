import numpy as np
import tensorflow as tf
import keras
import os
import matplotlib.pyplot as plt
from keras import losses, backend, models, layers
import keras_visualizer

LOAD_EXIST = os.path.exists("my_model.keras")
TRAINING = False

state_count = 4

if not LOAD_EXIST:
    # Define the model
    model = models.Sequential([
        layers.Dense(20, input_dim=3+state_count, activation='relu'),  # Hidden layer
        layers.Dense(3+state_count, activation='sigmoid')             # Output layer
    ])

    # Compile the model
    model.compile(optimizer='SGD', loss='mean_squared_error')

if LOAD_EXIST:
    model = keras.models.load_model("my_model.keras")

keras_visualizer.visualizer(model, settings={'MAX_NEURONS': 8})
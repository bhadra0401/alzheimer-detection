import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load trained MobileNet model
model = load_model("models/mobilenet_alzheimer.keras")

# Remove final classification layer
feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-2].output
)

# Data Generator
datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    "AugmentedAlzheimerDataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Extract Features
features = feature_extractor.predict(generator)

labels = generator.classes

os.makedirs("features", exist_ok=True)

# Save
np.save("features/X_features.npy", features)
np.save("features/y_labels.npy", labels)

print("Features Shape:", features.shape)
print("Labels Shape:", labels.shape)

print("Feature Extraction Complete!")
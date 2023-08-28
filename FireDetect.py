import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
fire_dir = './Train/Fire'
non_fire_dir = './Train/Non-Fire'
fire_files = [os.path.join(fire_dir, file) for file in os.listdir(fire_dir)]
non_fire_files = [os.path.join(non_fire_dir, file) for file in os.listdir(non_fire_dir)]
#fire_frames = [cv2.imread(file) for file in fire_files]
fire_frames = []
for file in fire_files:
    frame = cv2.imread(file)
    # resize the image to (224, 224, 3)
    frame = cv2.resize(frame, (224, 224))
    fire_frames.append(frame)
#non_fire_frames = [cv2.imread(file) for file in non_fire_files]
non_fire_frames=[]
for file in non_fire_files:
    frame=cv2.imread(file)
    if frame is not None:
        frame=cv2.resize(frame,(224,224))
        non_fire_frames.append(frame)
def preprocess_frames(frames):
    # Resize the frames to a fixed size
    resized=[cv2.resize(frame,(224,224)) for frame in frames]
    # Convert the frames to a normalized array
    normalized = np.array(resized) / 255.0
    # Return the preprocessed frames
    return normalized
X_fire = preprocess_frames(fire_frames)
X_non_fire = preprocess_frames(non_fire_frames)
y_fire = np.ones(len(X_fire))
y_non_fire = np.zeros(len(X_non_fire))
X_train, X_test, y_train, y_test = train_test_split(np.concatenate([X_fire, X_non_fire]), np.concatenate([y_fire, y_non_fire]), test_size=0.2)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=45, batch_size=32, validation_data=(X_test, y_test))
accuracy = model.evaluate(X_test, y_test)[1]
model.save('fire_detection.h5')

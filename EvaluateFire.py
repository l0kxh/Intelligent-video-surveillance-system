import matplotlib.pyplot as plt
def preprocess_frame(frame):
    # Resize the frame to a fixed size
    resized = cv2.resize(frame, (224, 224))
    # Convert the frame to a normalized array
    normalized = np.array(resized) / 255.0
    # Reshape the frame into a list
    preprocessed = np.expand_dims(normalized, axis=0)
    # Return the preprocessed frame
    return preprocessed
image_path = 'test6.png'
model=tf.keras.models.load_model('fire_detection.h5')
image = cv2.imread(image_path)
plt.imshow(image)
preprocessed_image = preprocess_frame(image)
prediction = model.predict(preprocessed_image)[0][0]
print(prediction)
if prediction >0.5:
    print("Fire")
else:
    print("No fire")

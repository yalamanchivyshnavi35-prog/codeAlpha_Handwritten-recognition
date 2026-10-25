
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the MNIST dataset
print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)

# 2. Preprocess the images
# Normalize pixel values from 0-255 to 0-1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension for CNN (28, 28, 1)
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# 3. Build the CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),

    layers.Dense(10, activation="softmax")
])

# 4. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train the model
print("\nTraining the CNN model...")

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# 6. Evaluate the model
test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# 7. Save the trained model
model.save("handwritten_digit_model.keras")
print("Model saved successfully!")

# 8. Predict a test image
index = 0

image = x_test[index]
prediction = model.predict(image[np.newaxis, ...], verbose=0)

predicted_digit = np.argmax(prediction)
actual_digit = y_test[index]

print("\nPredicted digit:", predicted_digit)
print("Actual digit:", actual_digit)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(
    f"Predicted: {predicted_digit} | Actual: {actual_digit}"
)
plt.axis("off")
plt.show()

# 9. Classification report
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

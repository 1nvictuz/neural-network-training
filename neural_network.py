
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

# Use a random seed for repeatable results
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load handwritten digit images
(x_data, y_data), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# 2. Normalize images between 0 and 1
x_data = x_data.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 3. Split training and validation data
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data,
    test_size=0.2,
    random_state=42,
    stratify=y_data
)

# 4. Create a simple neural network
model = Sequential([
    tf.keras.Input(shape=(28, 28)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")
])

# 5. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Train the model
model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=5,
    batch_size=32
)

# 7. Evaluate on test data
test_loss, test_accuracy = model.evaluate(
    x_test, y_test
)

print("\n=== FINAL RESULTS ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 8. Save the trained model
model.save("mnist_model.keras")

print("Model saved successfully!")

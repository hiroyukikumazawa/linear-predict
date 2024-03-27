import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Generate synthetic data: y = 2x + 1
X_train = np.random.rand(100, 1)  # 100 random points
y_train = 2 * X_train + 1 + np.random.rand(100, 1) * 0.1

# Generate some data for testing the model
X_test = np.array([[0.1], [0.5], [0.9]])


model = tf.keras.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=(1,), activation="linear")]
)

model.compile(optimizer="sgd", loss="mean_squared_error")

model.fit(X_train, y_train, epochs=50)


y_pred_train = model.predict(X_train)

plt.scatter(X_train, y_train, label="Training data")
plt.plot(X_train, y_pred_train, color="red", label="Model predictions")
plt.legend()
plt.show()

# Predicting new values
X_new = np.array([[0.1], [0.5], [0.9]])  # New input data
y_pred_new = model.predict(X_new)

print("Predictions for new data:", y_pred_new.flatten())
# Save the model
model.save("linear_regression_model.h5")

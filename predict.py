import numpy as np
import tensorflow as tf


def predict(values: list[list[float]]):
    model = tf.keras.models.load_model("./linear_regression_model.h5")
    print(values)
    X_new = np.array(values)
    y_pred = model.predict(X_new)
    # print("Predictions:", y_pred.flatten())
    np.savetxt("predictions.csv", y_pred, delimiter=",", header="Predictions")
    return y_pred.flatten()

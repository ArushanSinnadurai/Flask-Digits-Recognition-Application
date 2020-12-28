#loading libraries
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

# Model / data parameters
one_hot_encoding = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Gray scaling the images. [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "Number of training samples")
print(x_test.shape[0], "Number of testing samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, one_hot_encoding)
y_test = keras.utils.to_categorical(y_test, one_hot_encoding)

model = keras.Sequential(
    [
        #input shape (28, 28, 1)
        keras.Input(shape=input_shape),
        #filter = 32 
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        #pool to make 2d array of size 2x2
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        #Dropout is used to reduce overfitting
        layers.Dropout(0.5),
        #Output is 0-9
        layers.Dense(one_hot_encoding, activation="softmax"),
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.1)

scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
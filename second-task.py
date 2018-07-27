# NOTE: second-task_snippets.py is included just to prove that the network was trained
# also with different configurations (number of nodes, epochs)
# it is not for execution.

import cv2
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from helpers import resize_to_fit


SINGLE_LETTERS_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

data = []
labels = []
for image_file in paths.list_images(SINGLE_LETTERS_FOLDER):
    image = cv2.imread(image_file)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # set dimension for all images
    grayscale_image = resize_to_fit(grayscale_image, 20, 20)
    # need a third channel to work with keras
    grayscale_image = np.expand_dims(grayscale_image, axis=2)

    # add letter of image to labels
    label = image_file.split(os.path.sep)[-2]
    data.append(grayscale_image)
    labels.append(label)


# try scaling to improve training, float have higher precision
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters) into one-hot encodings
# we go from a "class representation" to a binary representation
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# save labels for third step
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

model = Sequential()
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(300, activation="relu"))
model.add(Dense(32, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
shift = 0.2
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=shift, height_shift_range=shift)
datagen.fit(X_train)
# Train the neural network
# toggle if you want to use image data generator or not
#history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train) / 32, epochs=10, verbose=1, validation_data=(X_test, Y_test), validation_steps=len(X_test) / 32)

# save the trained model for third step, we need it to predict unseen examples
model.save(MODEL_FILENAME)

#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()



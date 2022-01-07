import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense

image_dir = "datasets/"
dataset = []
label = []
no_image_tumor = os.listdir(image_dir + "no")
yes_image_tumor = os.listdir(image_dir + "yes")
INPUT_SIZE = 64

for i, image_name in enumerate(no_image_tumor):
    if image_name.split(".")[1] == 'jpg':
        images = cv2.imread(image_dir + "no/" + image_name)
        image = Image.fromarray(images, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_image_tumor):
    if image_name.split(".")[1] == 'jpg':
        images = cv2.imread(image_dir + "yes/" + image_name)
        image = Image.fromarray(images, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

X_train, x_test, Y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

X_train = normalize(X_train, axis=1)
x_test = normalize(x_test, axis=1)

# to_catgorical
Y_train = to_categorical(Y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
# model.add(Activation('sigmoid'))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, epochs=10, verbose=1, validation_data=(x_test, y_test), shuffle=False)

model.save("brain_tumor_model_cc.h5")




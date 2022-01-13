import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype(np.float32)/255
X_test = X_test.astype(np.float32)/255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

# model.summary()

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=4, verbose=1)

mc = ModelCheckpoint("bModel.h5", monitor='val_acc', verbose=1, save_best_only=True)

cb = [es, mc]

his = model.fit(X_train, y_train, epochs=5, validation_split=0.2, callbacks=cb)

model.save("bModel.h5")

''' model_S = keras.models.load_model("data/bModel.h5")

score = model_S.evaluate(X_test, y_test)

print(score) '''
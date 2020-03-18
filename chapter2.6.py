import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

from sklearn import datasets

iris = datasets.load_iris()
y_train = keras.utils.to_categorical(np.array([1. if x==0 else 0. for x in iris.target]), num_classes=2)
x_train = np.array([[x[2], x[3]] for x in iris.data])

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=2))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


model.fit(x_train, y_train,
          epochs=20,
          batch_size=20)
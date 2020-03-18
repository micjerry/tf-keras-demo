import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2

from sklearn import datasets

import matplotlib.pyplot as plt

iris = datasets.load_iris()
y_train = keras.utils.to_categorical(np.array([1. if x==0 else -1. for x in iris.target]), num_classes=2)
x_train = np.array([[x[0], x[3]] for x in iris.data])

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=2))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, kernel_regularizer=l2(0.01)))
model.add(Activation('linear'))  

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          epochs=40,
          batch_size=20)

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
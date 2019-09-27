from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adamax

model = Sequential()
model.add(Dense(32, input_shape=(13,), kernel_initializer='uniform', activation='softmax'))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))

model.compile(Adamax(lr=0.01),
      loss='categorical_crossentropy',
      metrics=['accuracy'])

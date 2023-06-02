from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPool2D


cnn3 = Sequential()
cnn3.add(Conv2D(8,kernel_size=(3,3), activation='relu', input_shape=(256,256,3)))
cnn3.add(MaxPool2D((2, 2)))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
cnn3.add(MaxPool2D(pool_size=(2,2)))
cnn3.add(Dropout(0.2))

cnn3.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
cnn3.add(MaxPool2D(pool_size=(2,2)))
cnn3.add(Dropout(0.2))

cnn3.add(Flatten())
cnn3.add(Dropout(0.2))
cnn3.add(Dense(32,activation='relu'))
cnn3.add(Dropout(0.2))
cnn3.add(Dense(1,activation='sigmoid'))

cnn3.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


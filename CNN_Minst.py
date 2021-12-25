from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import adadelta
from sklearn.metrics import confusion_matrix
import itertools
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
num_classes = 10
epochs =5
img_rows,img_cols = 28,28

(X_train , Y_train),(X_test,Y_test) = mnist.load_data()

if k.image_data_format() == "channels_first":
    X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
    X_test = X_test.reshape(X_test.shape[0],1,img_rows,img_cols)
    input.shape = (1,img_rows,img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
    X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)
    input_shape = (img_rows,img_cols,1)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /=255
X_test /=255
Y_train = to_categorical(Y_train,num_classes)
Y_test = to_categorical(Y_test,num_classes)

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=categorical_crossentropy,optimizer=adadelta(),metrics=['accuracy'])
h= model.fit(X_train , Y_train , batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,Y_test))
final_loss, final_acc = model.evaluate(X_test, Y_test, verbose=0)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
accuracy = h.history['acc']
val_accuracy = h.history['val_acc']
loss = h.history['loss']
val_loss = h.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

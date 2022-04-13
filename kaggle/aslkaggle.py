# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import scikitplot as skplt
import pandas as pd
import numpy as np 
import random 
import cv2 
import matplotlib.pyplot as plt 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 

train = pd.read_csv('/home/parth/Desktop/aslproject/kaggle/sign_mnist_train.csv')
test = pd.read_csv('/home/parth/Desktop/aslproject/kaggle/sign_mnist_test.csv')

# train_data = np.array(train, dtype='float32')

train_data = np.array(train, dtype='float32')
test_data = np.array(test, dtype='float32')

class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T', 'U', 'V', 'W', 'X', 'Y']


#Random sample sanity check 
i = random.randint(1, train.shape[0])
fig1, ax1 = plt.subplots(figsize=(10,10))
plt.imshow(train_data[i,1:].reshape((28,28)), cmap='gray')
plt.show(block=False)

print('Label for the image is: ', class_names[int(train_data[i,0])])

# Data distribution visualisation
fig = plt.figure(figsize=(18,18))
ax1 = fig.add_subplot(221)
train['label'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_title('Label')


# Normalize the Data between 0 to 1

X_train = train_data[:,1:]/255.
X_test = test_data[:,1:]/255.


y_train = train_data[:,0]
y_train_categorical = to_categorical(y_train, num_classes=25)

y_test = test_data[:,0]
y_test_categorical = to_categorical(y_test, num_classes=25)

# Reshape the array for NN 

X_train = X_train.reshape(X_train.shape[0], *(28,28,1))
X_test = X_test.reshape(X_test.shape[0], *(28,28,1))



## Model Creation 

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(25, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


result = model.fit(X_train, y_train_categorical, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test_categorical))

loss = result.history['loss']
val_loss = result.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.figure(figsize=(10,10))
plt.plot(epochs, loss, 'y', label= 'Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

acc = result.history['accuracy']
val_acc = result.history['val_accuracy']
plt.figure(figsize=(10,10))
plt.plot(epochs, acc, 'y', label= 'Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.grid(True)
plt.legend()
plt.show()


prediction = model.predict_classes(X_test)


prediction.history

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns 

accuracy_sc = accuracy_score(y_test, prediction)  
print('Accuracy Score is: ', accuracy_sc)

i = random.randint(1, len(prediction))
plt.imshow(X_test[i,:,:,:])
print("predicted label: ", class_names[int(prediction[i])])
print('True label:', class_names[int(y_test[i])])


fig,ax = plt.subplots(figsize=(25,25))
skplt.metrics.plot_confusion_matrix(y_test, prediction,
                                    normalize = True,
                                    title="Confusion Matrix",
                                    cmap="Oranges",
                                    ax=ax)



cm = confusion_matrix(y_test,prediction)

## Plot fractional incorrect misclassifications 

incorr_frac = 1 - np.diag(cm) / np.sum(cm, axis=1)
fig,ax = plt.subplots(figsize=(15,15))
plt.bar(np.arange(24), incorr_frac)
plt.xlabel('True label')
plt.ylabel('Fraction of incorrect predictions')
plt.xticks(np.arange(24))


# predictions = prediction.reshape(1,-1)[0]
print(classification_report(y_test, prediction))

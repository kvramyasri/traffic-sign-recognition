#import required libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Taking the input images and convert to numpy array
data=[]
class_labels=[]

img_height = 30
img_width = 30
img_channels = 3
num_classes = 43
n_inputs = img_height * img_width * img_channels

#Reading the trainig data
for n_class in range(num_classes) :
    path = "/Users/rams/Desktop/flaskapp/Train/{0}/".format(n_class)
    print(path)
    Class=os.listdir(path)
    for cls in Class:
        try:
            img=cv2.imread(path+cls)
            img_from_array = Image.fromarray(img, 'RGB')
            img_size = img_from_array.resize((img_height, img_width))
            data.append(np.array(img_size))
            class_labels.append(n_class)
        except AttributeError:
            print(" ")
            
data=np.array(data)
class_labels=np.array(class_labels)

#shuffle the images for better accuracy
s=np.arange(data.shape[0])
np.random.seed(43)
np.random.shuffle(s)
data=data[s]
class_labels=class_labels[s]

#split the data to training and validation sets
(X_train,X_val)=data[(int)(0.2*len(class_labels)):],data[:(int)(0.2*len(class_labels))]
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255
(y_train,y_val)=class_labels[(int)(0.2*len(class_labels)):],class_labels[:(int)(0.2*len(class_labels))]

#convert the labelled data to numerical data using one hot encoding
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

#create a sequential model and add conv2d,maxpooling layers
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='tanh', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='tanh'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='tanh'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
 
#create the learning phase for model using compile method
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

#Train the model with fit method and earlystopping as the callback function
epochs = 20
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs,
validation_data=(X_val, y_val))

#plot the graph for training accuracy and valdidation accuracy
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#plot the graph for training loss and valdidation loss
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#Test data prediction
y_test=pd.read_csv("/Users/rams/Desktop/flaskapp/Test.csv")
class_labels=y_test['Path'].to_numpy()
y_test=y_test['ClassId'].values

data=[]

for f in class_labels:
    img=cv2.imread('/Users/rams/Desktop/flaskapp/Test/'+f.replace('Test/', ''))
    img_from_array = Image.fromarray(img, 'RGB')
    img_size = img_from_array.resize((img_height, img_width))
    data.append(np.array(img_size))

X_test=np.array(data)
X_test = X_test.astype('float32')/255 
pred_arr = model.predict_classes(X_test) 
print("Accuracy Score: ",accuracy_score(y_test, pred_arr))
print(classification_report(y_test, pred_arr))

#Save the model in .h5 fromat
model.save('/Users/rams/Desktop/flaskapp/modelCNN.h5')
#import required libraries 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score



#Taking the input images and convert to numpy array
data=[]
class_labels=[]

img_height = 32
img_width = 32
img_channels = 3
n_classes = 43
n_inputs = img_height * img_width * img_channels

#Reading the trainig data
for n_cls in range(n_classes) :
    path = "/Users/rams/Desktop/flaskapp/Train/{0}/".format(n_cls)
    print(path)
    Class=os.listdir(path)
    for cls in Class:
        try:
            img=cv2.imread(path+cls)
            img_from_array = Image.fromarray(img, 'RGB')
            img_size = img_from_array.resize((img_height, img_width))
            data.append(np.array(img_size))
            class_labels.append(n_cls)
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


#using pre-traing VGG16 model with imagenet as weights
pre_trained_model = VGG16(
    include_top= False,
    weights="imagenet",
    input_shape= (32,32,3))

#set the trainable of layers to false
for layer in pre_trained_model.layers:
    layer.trainable = False

#get the last layer of the pretrained model
last_layer = pre_trained_model.get_layer('block5_pool')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

#generate the output layer
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(43, activation='softmax')(x)           

#create the model from input and output
model = Model(pre_trained_model.input, x) 


#create the learning phase for model using compile method
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

#Train the model with fit method and earlystopping as the callback function
epochs = 20
monitor=EarlyStopping(monitor='val_loss',min_delta=1e-3, patience=5, verbose=1, mode='auto',restore_best_weights=True) 
history = model.fit(X_train, y_train,  callbacks = [monitor],batch_size=32, epochs=epochs,
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
pred_arr = model.predict(X_test) 
pred=np.argmax(pred_arr,axis=1)
print(pred)

#Save the model in .h5 fromat
model.save('/Users/rams/Desktop/flaskapp/modelvgg16.h5')
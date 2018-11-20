#!/usr/bin/python
import tensorflow as tf
import numpy as np
import pandas
import os
import argparse
import sys
import cv2
from PIL import Image
import PIL
import cv2
from tensorflow.keras.layers import Dense, Dropout,Flatten,Activation,Conv2D, MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16


def readCsv(filename):

    res = pandas.read_csv(filename)
    np.save("res",res)

    print(type(res))
    return res

def dataset(res,num):
    y_train, x_train, x_test, y_test = [],[],[],[]
    
    for i in range(0,len(res)):
        try:
            emotion = res.iloc[i][0]
            img = res.iloc[i][1]
            usage = res.iloc[i][2]
            val = img.split(" ")
            
            
            pixels = np.array(val, "float32")
            
            
            emotion = tf.keras.utils.to_categorical(emotion,num_classes=num)
            
            if "Training" in usage:
                y_train.append(emotion)
                x_train.append(pixels)
            if "PublicTest" in usage:
                y_test.append(emotion)
                x_test.append(pixels)
        except Exception as e:
            pass
    
    x_train = np.array(x_train,'float32')
    y_train = np.array(y_train,'float32')
    x_test = np.array(x_test,'float32')
    y_test = np.array(y_test,'float32')
    #normalising the data
    x_train /= 255
    x_test /= 255
    np.save("x_train",x_train)
    np.save("y_train",y_train)
    np.save("x_test",x_test)
    np.save("y_test",y_test)

    return (x_train,y_train,x_test,y_test)
def CNN_model2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=96, kernel_size=(3,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=96, kernel_size=(3,3), strides=2))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=192, kernel_size=(3,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters=192, kernel_size=(3,3), strides=2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation="softmax"))
    return model
    

def vggModel(input_shape,num_classes):
    VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    topLayerModel = Sequential()
    topLayerModel.add(Dense(256, input_shape=(512,), activation='relu'))
    topLayerModel.add(Dense(256, input_shape=(256,), activation='relu'))
    topLayerModel.add(Dropout(0.5))
    topLayerModel.add(Dense(128, input_shape=(256,), activation='relu'))
    topLayerModel.add(Dense(num_classes, activation='softmax'))
    return topLayerModel
    
    
def CNN_model1(input_shape,num_classes):
    #3 layers model
    model = Sequential()
    #layer 1
    model.add(Conv2D(64, (5,5),input_shape=input_shape, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #layer 2
    model.add(Conv2D(128, (3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    #layer 3
    model.add(Conv2D(128, (3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'  ))
    model.add(Dropout(0.1))
    
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    
    return model
    
def compile_model(model,x,y,x_test,y_test):
    
    #compile the model with preset parameters
    model.compile(loss='categorical_crossentropy', optimizer='adam' ,metrics=["accuracy"])
    



    

if __name__ == "__main__":
    #args
    parser = argparse.ArgumentParser(description='Model parameters')
    parser.add_argument('-b','--batch-size',help="Number of batch size",default="128")
    parser.add_argument('-e','--epochs',help="Number of epochs",default="15")
    args = parser.parse_args() 
    batch_size = args.batch_size
    epochs = args.epochs
    
    #number of labels (7 emotions)
    num_classes = 7
    IMG_SIZE = 48
    
    filename = "fer2013.csv"
    #if the csv array is not present call the method to read the csv otherwise load it
    if os.path.isfile("res.npy"):
        print("file found")
        res = np.load("res.npy")
        print(type(res))
    else:
        
        print("reading csv../")
        res=readCsv(filename)

    
    #initialisation of the data arrays
    x_train, y_train, x_test, y_test = [],[],[],[]
    #if the data arrays are present load them, otherwise call the dataset method
    if os.path.isfile("x_train.npy"):
        print("datasets found")
        x_train = np.load("x_train.npy")
        y_train = np.load("y_train.npy")
        x_test = np.load("x_test.npy")
        y_test = np.load("y_test.npy")
        

        
    else:
        print("sa")
        x_train,y_train,x_test,y_test = dataset(res,num_classes)

    
    #reshape the pictures array to get a rank 4 array (length, 48, 48, 1)
    x_train = x_train.reshape(x_train.shape[0],IMG_SIZE,IMG_SIZE,1)
    x_test = x_test.reshape(x_test.shape[0],IMG_SIZE,IMG_SIZE,1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    

    #info on the data
    
    print("pictures array : ",x_train.shape,type(x_train))
    print("shape of the labels array",y_train.shape,type(y_train))

    #load the model (convolutional neural network 3 layers)
    model = CNN_model1((48,48,1),num_classes)
    #compiling the model
    compile_model(model,x_train,y_train,x_test,y_test)
    

    #save the model
    if os.path.isfile('fer.h5'):
        model.load_model('fer.h5') #load weights
    #train the model, the save it
    else:
        model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_train,y_train),epochs=epochs)
        model.save('fer.h5')
        
        
    
       
    
    
    

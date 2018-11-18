import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os.path
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
    x_train /= 255
    x_test /= 255
    np.save("x_train",x_train)
    np.save("y_train",y_train)
    np.save("x_test",x_test)
    np.save("y_test",y_test)

    return (x_train,y_train,x_test,y_test)
def evenShittierModel(input_shape, num_classes):
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
    
    
def shittyModel(input_shape,num_classes):
    model = Sequential()
    # model.add(tf.layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
    # model.add(tf.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    # #2nd convolution layer
    # model.add(tf.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    
    # #3rd convolution layer
    # model.add(tf.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.layers.MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

    # model.add(tf.layers.Flatten())

    # #fully connected neural networks
    # model.add(tf.layers.Dense(64))
    # model.add(tf.layers.Dense(num_classes, activation='softmax'))
    model.add(Conv2D(64, (5,5),input_shape=input_shape, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(128, (3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'  ))
    model.add(Dropout(0.1))
    
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    
    return model
    
def train(model,x,y,x_test,y_test):
    #model.compile(optimizer='adam',
    #loss='categorical_crossentropy', 
    #metrics=['accuracy'])
    
    ####
    #gen = ImageDataGenerator()
    #train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
    #rms = RMSprop()
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
    #####
    model.compile(loss='categorical_crossentropy', optimizer='adam' ,metrics=["accuracy"])
    



    

if __name__ == "__main__":
    
    batch_size = 156
    num_classes = 7
    IMG_SIZE = 48
    filename = "fer2013.csv"
    if os.path.isfile("res.npy"):
        print("file found")
        res = np.load("res.npy")
        print(type(res))
    else:
        res=readCsv(filename)

    x_train, y_train, x_test, y_test = [],[],[],[]
    
    if os.path.isfile("x_train.npy"):
        print("datasets found")
        x_train = np.load("x_train.npy")
        y_train = np.load("y_train.npy")
        x_test = np.load("x_test.npy")
        y_test = np.load("y_test.npy")
        

        
    else:
        print("sa")
        x_train,y_train,x_test,y_test = dataset(res,num_classes)

    print(type(x_train))
    x_train = x_train.reshape(x_train.shape[0],IMG_SIZE,IMG_SIZE,1)
    x_test = x_test.reshape(x_test.shape[0],IMG_SIZE,IMG_SIZE,1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print(x_train.shape)
    print(y_train)
    print(x_test,y_test)

    model = evenShittierModel((48,48,1),num_classes)
    
    train(model,x_train,y_train,x_test,y_test)
    

    
    if os.path.isfile('fer.h5'):
        model.load_model('fer.h5') #load weights

    else:
        model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_train,y_train),epochs=3)
        model.save('fer.h5')
        
        
    
       
    
    
    

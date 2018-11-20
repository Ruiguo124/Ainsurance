# Ainsurance: Project for McGill's 2018 Code Jam Hackathon. 
# Won first place for Computer Vision!!!
![image_test](https://github.com/Ruiguo124/Ainsurance/blob/master/gallery.jpg)

## Usage:
The model is already trained and saved in the fer.h5. 

If you want to train your own model you firstly need to download the facial expression recognition dataset from kaggle [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

Must need python 3.6, tensorflow and opencv3

To compile and fit the model

```console
./main.py --batch-size=<nb> --epochs=<nb>
```
  
To run the live video feed facial expression recognition
```console
./videocapture --filname=<model_filename.extension>
```


## Inspiration: 

It has been proved by various research papers that depression and sadness can take its toll on someone's physical health. Not only does it increase your chances of getting sick, it can also reduce your lifespan by up to 10 years. _ (Canadian Medical Association Journal: http://www.cmaj.ca/content/189/42/E1304)_

## AInsurance's uses: 

There are various ways in which insurance premiums are calculated. Tobacco users are usually charged 1.5x more than non smokers, even though depression has higher toll on someone's health, no insurance premiums could be calculated until NOW.

AInsurance's emotion detection can also help the insurance seller negotiate his prices. Since our AI can detect anger, sadness, happiness, disgust, and fear. The AI detects if the insurance premium's price is too high or not high enough, and the insurance seller could negotiate better.

AInsurance helps you gather the data and process it in a way you never knew you needed!

## Implementation: 

We have implemented a multiple-layer convolutional neural network which detects someone's emotion in real time via video feed and uses a basic algorithm to calculate insurance price using the insurance interview video. We then feed 30 thousand 48x48 pixels pictures through the network for 15 epochs. The model returns approximately 68% accuracy. A total of 7 emotions can be detected (Sad, Happy, Angry, Disgust, Fear, Neutral and Surprised)

We used: Python Tensorflow Opencv-python Keras

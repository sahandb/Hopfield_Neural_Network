# Hopfield_Neural_Network
I designed a parallel Hopfield neural network to perform normal facial image recognition

Through facial expression images by using a dataset (Yale Face Database) which contains 165 grayscale images of 15 different people. We have 11 images per person (subject), one per each facial expression or configuration: center-light, w/glasses, happy, left-light, w/no glasses, normal, rightlight, sad, sleepy, surprised, and winked. Here for this project, the first 7 subjects have been chosen.

At first I read the all images from train and test then turn them from RGB to gray and resize the images from 243x320 to 32x32 pixels and then reshape each image into a single column vector and then turn the 1024 image vector to 8bit binary representation and do that with all 7 images

After that I train 8 parallel Hopfield network for each column of the resulted images. Each network will have 1024 inputs and 1024 outputs. For training the first network, use the first column of each matrix and the second and …


After that I test the trained nets with test data and get each column to each net and get result and after that stick the 1024x70 of each column together and make 1024x70 8bit test and after that turn that to 70 image 32x32

And check the accuracy at the end

I use pre defined library for Hopefield in python (DHNN)

![image](https://user-images.githubusercontent.com/24508376/219622053-9978b203-dc21-4d7c-b644-bce39f6dd819.png)

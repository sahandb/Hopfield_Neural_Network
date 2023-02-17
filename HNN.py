# imports
import numpy as np
import pandas as pd
import cv2
import glob
from matplotlib import pyplot as plt
from dhnn import DHNN
import matplotlib.gridspec as gridspec
from PIL import Image

# routes of train and test
dirTrain = './yaleface/TrainData/*.*'
dirTest = './yaleface/TestData/*.*'


def preProcessImage(location):
    resizedImg = list()
    image8bits = list()
    imageSubject = list()
    for filename in glob.glob(location):
        image = Image.open(filename).convert('L')  # convert ot grayscale
        resizedImage = cv2.resize(np.array(image), (32, 32))  # resize images
        resizedImg.append(resizedImage)
        flattenImage = np.array(resizedImage).flatten()
        image8bit = list(map(lambda x: list('{0:08b}'.format(x)), flattenImage)) #make images to8 bit
        image8bits.append(np.array(image8bit, dtype=np.ubyte))
        imageSubject.append(filename.split("\\")[1])  # subject File names
    image8bits = np.where(np.array(image8bits) == 0, -1, 1)  # turn image to -0 and 1
    return imageSubject, np.array(image8bits), np.array(resizedImg)


def postProcessImage(images):
    images = np.where(np.array(images) == -1, 0, 1)
    i, j, k = images.shape
    flattenImage = np.empty((i, j))
    for x in range(i):
        for y in range(j):
            flattenImage[x][y] = (int(''.join(str(g) for g in images[x, y]), 2))  # restore image flatten image
        # image[:, :, i] = map(lambda x: int(x[:, :, x], 2), images)
    retrievedImage = flattenImage.reshape((70, 32, 32))  # reshape images to 70 img with 32 point 32

    return retrievedImage


def predictClass(namesTr, namesTe, retImg, orgTrainImg):
    classPredict = []
    for labelTest in range(len(namesTe)):
        minDiff = []
        for labelTrain in range(len(namesTr)):
            f = np.linalg.norm(orgTrainImg[labelTrain] - retImg[labelTest])
            minDiff.append(f)
        classPredict.append(namesTr[np.argmin(minDiff)])
    return classPredict


namesTrain, matrixTrain, resizedTrain = preProcessImage(dirTrain)
namesTest, matrixTest, resizedTest = preProcessImage(dirTest)
# train network with 8 hatfield net
train1 = matrixTrain[:, :, 0]
test1 = matrixTest[:, :, 0]
model1 = DHNN()
model1.train(train1)
recovery1 = model1.predict(test1)

train2 = matrixTrain[:, :, 1]
test2 = matrixTest[:, :, 1]
model2 = DHNN()
model2.train(train2)
recovery2 = model2.predict(test2)

train3 = matrixTrain[:, :, 2]
test3 = matrixTest[:, :, 2]
model3 = DHNN()
model3.train(train3)
recovery3 = model3.predict(test3)

train4 = matrixTrain[:, :, 3]
test4 = matrixTest[:, :, 3]
model4 = DHNN()
model4.train(train4)
recovery4 = model4.predict(test4)

train5 = matrixTrain[:, :, 4]
test5 = matrixTest[:, :, 4]
model5 = DHNN()
model5.train(train1)
recovery5 = model5.predict(test5)

train6 = matrixTrain[:, :, 5]
test6 = matrixTest[:, :, 5]
model6 = DHNN()
model6.train(train6)
recovery6 = model6.predict(test6)

train7 = matrixTrain[:, :, 6]
test7 = matrixTest[:, :, 6]
model7 = DHNN()
model7.train(train7)
recovery7 = model7.predict(test7)

train8 = matrixTrain[:, :, 7]
test8 = matrixTest[:, :, 7]
model8 = DHNN()
model8.train(train8)
recovery8 = model8.predict(test8)

testImages = np.stack((recovery1, recovery2, recovery3, recovery4, recovery5, recovery6, recovery7, recovery8), axis=-1)
retrievedImg = postProcessImage(testImages)

namesPredict = predictClass(namesTrain, namesTest, retrievedImg, resizedTrain)

accuracy = 0
for n in range(len(namesTest)):
    if namesTest[n].split(".")[0] == namesPredict[n].split(".")[0]:
        accuracy += 1
print((accuracy / len(namesTest)) * 100)

dictionary = dict()
for counter in range(7):
    dictionary[namesTrain[counter].split(".")[0]] = resizedTrain[counter]

fig = plt.figure(figsize=(10, 30))
fig.set_size_inches(5, 60)
for fi in range(len(namesTest)):
    fig.add_subplot(70, 3, fi + 1)
    plt.title(namesTest[fi].split(".")[0])
    plt.imshow(resizedTest[fi], cmap='gray', vmin=0, vmax=255, aspect='equal')
    plt.title(namesTest[fi].split(".")[0])
    plt.imshow(dictionary[namesTest[fi].split(".")[0]], cmap='gray', vmin=0, vmax=255, aspect='equal')
    plt.title(namesPredict[fi].split(".")[0])
    plt.imshow(retrievedImg[fi], cmap='gray', vmin=0, vmax=255, aspect='equal')
plt.show()

# fig, big_axes = plt.subplots( figsize=(20, 80) , nrows=70, ncols=1, sharey=True)
#
# for row, big_ax in enumerate(big_axes, start=1):
#     big_ax.set_title("Subplot row %s \n" % row, fontsize=16)
#
#     # Turn off axis lines and ticks of the big subplot
#     # obs alpha is 0 in RGBA string!
#     big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
#     # removes the white frame
#     big_ax._frameon = False
#
# for ffi in range(len(namesTest)):
#     ax = fig.add_subplot(70,3,ffi+1)
#     ax.set_title('Plot title ' + str(ffi))
#
# fig.set_facecolor('w')
# plt.tight_layout()
# plt.show()

# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import os
import glob
import cv2

def flattenResizeImage(image, size=(128, 128)):
	return cv2.resize(image, size).flatten()

def main():
    #Replace your data path here
    datapath = 'C:\\Users\\prana\\Downloads\\data'
    imagelist = []
    labels = []
    print("Read data")
    for filename in glob.glob(datapath + '\\*\\*.jpg'):
        img = cv2.imread(filename)
        pixels = flattenResizeImage(img)
        imagelist.append(pixels)
        head, tail = os.path.split(filename)
        headtail, tailtail = os.path.split(head)
        #Add labels of all images
        labels.append(tailtail)
    print("done reading data")

    #To keep mapping of labels to their corresponding integer values
    intlist = []
    labelist = []
    print("Split data")
    (trainRI, testRI, trainRL, testRL) = train_test_split(imagelist, labels, test_size=0.30, random_state=42)
    print("done spliting data")
    print("Train Kmeans classifier")
    kmeans = KMeans(init='random',n_clusters=256) #vary the number of clusters according to the data we feed in
    print("fit data")
    kmeans.fit(trainRI)
    print("Predict the labels for test data")
    predicted = kmeans.predict(testRI)

    print("Associate labels with integers")
    trainingerror = 0
    trainingaccuracy = 0;
    traininglabel = np.asarray(trainRL)
    for x in range(0,len(kmeans.labels_)-1):
        if(kmeans.labels_[x] not in intlist):
            if(traininglabel[x] not in labelist):
                intlist.append(kmeans.labels_[x])
                labelist.append(traininglabel[x])
                trainingaccuracy = trainingaccuracy + 1
            else:
                if (kmeans.labels_[x] in intlist) & (traininglabel[x] in labelist):
                    trainingaccuracy = trainingaccuracy + 1
                else:
                    trainingerror = trainingerror + 1
        else:
            if (kmeans.labels_[x] in intlist) & (traininglabel[x]  in labelist):
                trainingaccuracy = trainingaccuracy + 1
            else:
                trainingerror = trainingerror + 1

    #print(intlist)
    #print(labelist)

    testinglabel = np.asarray(testRL)
    testingerror = 0
    testaccuracy = 0
    for y in range(0,len(predicted)-1):
        if testinglabel[y] in labelist:
            testval = intlist[labelist.index(testinglabel[y])]
            if testval != predicted[y]:
                testingerror = testingerror + 1
            else:
                testaccuracy = testaccuracy + 1
        else:
            testaccuracy = testaccuracy + 1

    trainingaccuracy = trainingaccuracy/len(kmeans.labels_)
    testingaccuracy  = testaccuracy/len(predicted)
    print("traing accuracy ")
    print(trainingaccuracy)
    print("testing accuracy ")
    print(testingaccuracy)
    print("done with kmeans!")

if __name__== "__main__":
	main()





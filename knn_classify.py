 from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def extract_color_histogram(image, bins=(32, 32, 32)):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
     [0, 180, 0, 256, 0, 256])
  if imutils.is_cv2():
     hist = cv2.normalize(hist)
  else:
     cv2.normalize(hist, hist)
  return hist.flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
  help="input image directory")
ap.add_argument("-k", "--neighbors", type=int, default=1,
  help="number of neighbors")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawPixels = []
features = []
labels = []

# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
  image = cv2.imread(imagePath)
  label = imagePath.split(os.path.sep)[1].split(".")[-1]
  size = (128, 128)
  pixels = cv2.resize(image, size).flatten()
  hist = extract_color_histogram(image)
  rawPixels.append(pixels)
  features.append(hist)
  labels.append(label)
  if i > 0 and ((i + 1)% 200 == 0 or i ==len(imagePaths)-1):
     print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))


rawPixels = np.array(rawPixels)
features = np.array(features)
labels = np.array(labels)

(trainRP, testRP, trainRL, testRL) = train_test_split(
  rawPixels, labels, test_size=0.30, random_state=35)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
  features, labels, test_size=0.30, random_state=35)

model = KNeighborsClassifier(n_neighbors=args["neighbors"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("accuracy: {:.2f}%".format(acc * 100))






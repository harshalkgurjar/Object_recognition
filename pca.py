import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os

#PCA with image reconstruction
def pca(img):
    try:
        #Mean centered convariance matrix
        covariance_matrix = img - np.mean(img , axis = 0)
        #Finding eigenValues and Vectors
        eigenValues, eigenVectors = np.linalg.eigh(np.cov(covariance_matrix))
        eigenVecSize = np.size(eigenVectors, axis =0)
        #Sort eigenvalues in descending order
        index = np.argsort(eigenValues)
        index = index[::-1]
        eigenVectors = eigenVectors[:,index]
        eigenValues = eigenValues[index]
        #Number of principal components to be used for reconstruction of image
        numOfPC = 75
        if numOfPC <eigenVecSize or numOfPC >0:
            eigenVectors = eigenVectors[:, range(numOfPC)]
        #Reconstruction of image using the covariance matrix and the eigen vectors
        reconstructed = np.dot(eigenVectors.T, covariance_matrix)
        reconstructedMeanAdjusted = np.dot(eigenVectors, reconstructed) + np.mean(img, axis = 0).T
        reconstructedImageMatrix = np.uint8(np.absolute(reconstructedMeanAdjusted))
        return reconstructedImageMatrix
    except:
        print("")


def main():
    #Reading all images one by one
    datapath = 'C:\\Users\\prana\\Downloads\\data'
    outputpath = "C:\\Users\\prana\\Downloads\\recon\\"
    for filename in glob.glob(datapath + '\\*\\*.jpg'):
        try:
            img = scipy.ndimage.imread(filename)
            head, tail = os.path.split(filename)
            #print(head)
            #print(tail)
            headtail, tailtail = os.path.split(head)
            #print(headtail)
            #print(tailtail)
            imgMatrix = np.array(img)
            if(len(imgMatrix.shape) > 2):
                #for 3-d images - RGB color images
                imgR = imgMatrix[:,:,0]
                imgG = imgMatrix[:,:,1]
                imgB = imgMatrix[:,:,2]
                imgRReconstructed, imgGReconstructed, imgBReconstructed = pca(imgR), pca(imgG), pca(imgB)
                reconstructedImg = np.dstack((imgRReconstructed, imgGReconstructed, imgBReconstructed))
                reconstructedImg = Image.fromarray(reconstructedImg)
                try:
                    os.stat(outputpath + tailtail)
                except:
                    os.mkdir(outputpath + tailtail)
                scipy.misc.imsave(outputpath + tailtail + "\\" + tail, reconstructedImg)
            else:
                #for 2-d images
                imgW = imgMatrix[:, 0]
                imgB = imgMatrix[:, 1]
                imgWReconstructed, imgBReconstructed = pca(imgW), pca(imgB)
                reconstructedImg = np.dstack((imgWReconstructed, imgBReconstructed))
                reconstructedImg = Image.fromarray(reconstructedImg)
                try:
                    os.stat(outputpath + tailtail)
                except:
                    os.mkdir(outputpath + tailtail)
                scipy.misc.imsave(outputpath + tailtail  + "\\" + tail, reconstructedImg)
        except:
            print('')
    print("Done with PCA!")

if __name__== "__main__":
  main()

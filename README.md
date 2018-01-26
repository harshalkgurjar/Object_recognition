# ALDA- Object Recognition
# P22

This repository is code base for CSC 522 ALDA final project codes. 
1. Saloni Desai (sndesai)
2. FNU Manushri (manush)
3. Pranav Firake (ppfirake)
4. Harshal Gurjar (hkgurjar)

## Project idea: 

Object recognition is a process for identifying a specific object in a digital image or video. We intend to train our models using learning algorithms that rely on matching or pattern recognition using appearance-based or feature-based techniques. Our baseline model will be K-Nearest Neighbors (discriminative nearest neighbor classification model) which is a supervised learning technique. We will experiment with different values of K to figure out the best supervised model. We also intend to explore K-means clustering and convolutional neural network to build our object recognition system with the hypothesis that these unsupervised techniques (K-means and CNN) perform better than supervised techniques (KNN) in case of object recognition. We will be using accuracy to evaluate our models.

## DataSet: 

We will be using CALTECH 256 Dataset. It contains images of 256 object categories taken at varying
orientations, lighting conditions and backgrounds. It has a total of 30608 pictures with an average of
119 pictures in each category.
Data set: CALTECH 256 Image Dataset

http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar

## Technology or Packages used / required

- Ubuntu/Linux preferable
- python 2.7 or up
- Tensorflow 1.3.0
- Pillow : version 4.3.0
- glob
- random
- numpy : version 1.13.3 
- scipy : version 1.0.0
- sklearn 0.19.3
- OpenCV_python cv2 version: 3.0.0
- GPU preferable
- for CNN, we ran code on NCSU ARC - hpcc cluster with titanx reservation

For running these codes with more categories of input data please use high availability ARC cluster with large internal memory and preferably GPU support.

## Codes: 

### 1.PCA

Run it as ```python pca.py```
Kindly edit image data path before running code

### 2.KNN

Run it as ```python knn_classify.py --d "dataset_directory_name" --k number_of_k```
Kindly put image data path before running code in the same folder

### 3.Kmeans

Run it as ```python pca.py```
Kindly edit image data path in the script before running code

### 4.CNN

Run it as ```python cnn.py```
Kindly edit image data path in the script before running code

## Presentation : 

Please find presentation at 
https://github.ncsu.edu/ppfirake/alda_object_recognition/blob/master/1_Nov28th_11_P22.pdf
 
## Step wise guide for environment setup
Please perform the below steps in your host machine, if you wish to execute the above python scripts.
Ensure that your machine has enough memory, otherwise it may result in "Memory Error" due to high input size.
Language: Python 2.7 on an Ubuntu 16.04 machine
Pip Version: 8.1.1

Installation:
On an Ubuntu 16.04 machine, install Python, pip and virtualenv
Update and Upgrade: sudo apt-get update
Install pip and virtualenv: sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev
Create a virtualenv: virtualenv --system-site-packages -p python ~/keras-tf-venv

Install TensorFlow: 
Access the virtualenv: source ~/keras-tf-venv/bin/activate 
Install TensorFlow: pip install --upgrade tensorflow 

Install Keras and its dependencies:
pip install numpy scipy
pip install scikit-learn
pip install pillow
pip install h5py
pip install keras

Link Referred for Installation: http://deeplearning.lipingyang.org/2017/08/01/install-keras-with-tensorflow-backend/




## References

1. SVM-KNN: Discriminative Nearest Neighbor Classification for Visual Category Recognition (svm-knn)

2. Efficient clustering and matching for object class recognition (cluster)

3. Recurrent Convolutional Neural Network for Object Recognition. (CNN) 

4. CNN  model construction reference http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/ 

5. K means model reference http://ieeexplore.ieee.org/document/5702000/ 

6. KNN model reference: https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/



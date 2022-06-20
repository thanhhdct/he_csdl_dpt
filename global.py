
# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import glob
from scipy.spatial import distance

# fixed-sizes for image
fixed_size = tuple((500, 500))

# path to training data
train_path = "dataset/Image"

# bins for histogram
bins = 8

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    # cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# get the training labels
train_labels = os.listdir(train_path)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

i, j = 0, 0
k = 0

# num of images per class
images_per_class = 20

index_images = []

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    k = 1
    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"
        index_images.append(file)

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print ("[STATUS] processed folder: {}".format(current_label))
    j += 1

print ("[STATUS] completed Global Feature Extraction...")

# normalize the feature vector in the range (0-1)
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)

###############################################################
# extract features of query image
###########################################################

# path to test data
test_path = "dataset/test"

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file) 

    # resize the image
    image = cv2.resize(image, fixed_size)

    cv2.imshow("query image", image)
    cv2.waitKey(0) 
  
    #closing all open windows 
    cv2.destroyAllWindows()

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick  = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    ###################################
    # Concatenate global feature
    ###################################
    global_feature1 = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # normalize the feature vector in the range (0-1)
    global_feature2 = (global_feature1 - np.min(global_feature1))/np.ptp(global_feature1)

# calculate euclid distance: measures similarity between 2 feature matrices
results = []
for x in range(0, len(rescaled_features)):
    d = distance.euclidean(global_feature2, rescaled_features[x])
    results.append(d)

results1 = np.zeros(120, 2)
for x in range(0, len(rescaled_features)):
    d = distance.euclidean(global_feature2, rescaled_features[x])
    results1[x][0] = d
    results1[x][1] = x

results2 = sorted(results1, key = 0)

index = []
for i in range(0, len(rescaled_features)):
    index.append(i)

for i in range(0, len(rescaled_features)):
    for j in range(0, len(rescaled_features)-1):
        if(results[i] < results[j]):
            temp1 = results[i]
            results[i] = results[j]
            results[j] = temp1
            temp2 = index[i]
            index[i] = index[j]
            index[j] = temp2

print("results........................", results)
print("index.........................", index)
print("result2...........................", results2)

# for x in range(0,5):
#     image = cv2.imread(index_images[index[x]])
#     image = cv2.resize(image, fixed_size)
#     cv2.imshow("Searching image",image)
#     cv2.waitKey(0) 
  
#     #closing all open windows 
#     cv2.destroyAllWindows() 

for x in range(0,5):
    image = cv2.imread(index_images[results2[x][1]])
    image = cv2.resize(image, fixed_size)
    cv2.imshow("Searching image",image)
    cv2.waitKey(0) 
  
    #closing all open windows 
    cv2.destroyAllWindows() 
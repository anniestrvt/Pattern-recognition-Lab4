import cv2
import os
import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import distance
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import random
from xgboost import XGBClassifier


def load_images_from_folder(folder):
    images = list()
    labels = list()
    i = 0
    for filename in os.listdir(folder):
        if filename !='.DS_Store':
            path = folder + "/" + filename
            for cat in os.listdir(path):
                img = cv2.imread(path + "/" + cat,0)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if img is not None:
                    images.append(img)
                    labels.append(i)
            i = i + 1
    return images, labels


def load_images_class(folder):
    images = list()
    for cat in os.listdir(folder):
        img = cv2.imread(folder + "/" + cat,0)
        if img is not None:
            images.append(img)
    images_train, images_test = sklearn.model_selection.train_test_split(images, test_size = 50, random_state = 42)
    return images_train, images_test


def descriptor_features(X):
    descriptor_list = []
    akaze = cv2.AKAZE_create()
    for i in range(0, len(X)):
        kp,des = akaze.detectAndCompute(X[i], None)
        descriptor_list.extend(des)
    return descriptor_list


def kmeans(k, descriptor_list):
    kmeans = MiniBatchKMeans(n_clusters=k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_
    return visual_words

def find_index(image, center):
    count = 0
    ind = 0
    dist = 0
    for i in range(len(center)):
        if(i == 0):
            count = distance.euclidean(image, center[i])
            dist = count
            #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
        if(dist < count):
            ind = i
            count = dist
    return ind


def image_class(X, centers):
    dict_feature = list()
    akaze = cv2.AKAZE_create()
    for i in range(0, len(X)):
        print(i)
        kp, des = akaze.detectAndCompute(X[i], None)
        histogram = np.zeros(len(centers))
        for each_feature in des:
            ind = find_index(each_feature, centers)
            histogram[ind] += 1
        dict_feature.append(histogram)
    return dict_feature


X, y = load_images_from_folder('Photos')


descriptors = descriptor_features(X)
words = kmeans(200, descriptors)
X_new = image_class(X, words)






model = XGBClassifier()
arr = np.array(y)
arr1 = np.array(X_new)
model.fit(arr1, arr)
pickle.dump(model, open("pima.pickle.dat", "wb"))
pickle.dump(words, open("words.pickle.dat", "wb"))
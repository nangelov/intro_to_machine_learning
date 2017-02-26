#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
print "Decision Trees"
"""
from class_vis import prettyPicture, output_image

from prep_terrain_data import makeTerrainData
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl



prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
"""
print(len(features_train[0]))
from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print(accuracy)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
clf.predict(features_test)
print "predition time:", round(time()-t0, 3), "s"

#########################################################



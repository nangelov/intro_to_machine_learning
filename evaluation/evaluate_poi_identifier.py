#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn import cross_validation
from sklearn import tree
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn import metrics
print metrics.precision_score(labels_test,pred)
print metrics.recall_score(labels_test,pred)
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
true_negative = 0
false_negative = 0
true_positive = 0
false_positive = 0
for i in range(len(predictions)):
    if predictions[i] == 0 and true_labels[i] == 1:
        false_negative +=1
    if predictions[i] == 1 and true_labels[i] == 0:
        false_positive +=1
    if predictions[i] == 0 and true_labels[i] == 0:
        true_negative +=1
    if predictions[i] == 1 and true_labels[i] == 1:
        true_positive +=1

precision = float(true_positive)/(float(true_positive)+float(false_positive))
recall = float(true_positive)/(float(true_positive)+float(false_negative))
print precision
print recall

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)
print(accuracy)



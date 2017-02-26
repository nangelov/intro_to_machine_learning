#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from time import time

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','from_poi_to_this_person_emails', 'from_this_person_to_poi_emails'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

"""
### plot the data

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
"""

outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val != 'NaN':
        outliers.append((key, int(val)))


top_outliers = (sorted(outliers,key=lambda outlier:outlier[1],reverse=True)[:5])


### Task 3: Create new feature(s)
### new features are: from_poi_to_this_person_emails,from_poi_to_this_person_emails

def create_new_feature(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
from_poi_to_this_person_emails=create_new_feature("from_poi_to_this_person","to_messages")
from_this_person_to_poi_emails=create_new_feature("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["from_poi_to_this_person_emails"]=from_poi_to_this_person_emails[count]
    data_dict[i]["from_this_person_to_poi_emails"]=from_this_person_to_poi_emails[count]
    count +=1


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


"""
### plot the new features
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( from_poi_to_this_person_emails, from_this_person_to_poi_emails )

matplotlib.pyplot.xlabel("from_poi_to_this_person_emails")
matplotlib.pyplot.ylabel("from_this_person_to_poi_emails")
matplotlib.pyplot.show()
"""

### Task 4: Try a varity of classifiers

# Provided to give you a starting point. Try a variety of classifiers.
### from sklearn.naive_bayes import GaussianNB
### clf = GaussianNB()

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)


t0 = time()
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "Decision tree algorithm time before tuning:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test,pred)

print "accuracy before tuning ", accuracy





### Task 5: Tune your classifier to achieve better than .3 precision and recall 

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)


clf = tree.DecisionTreeClassifier(min_samples_split=8)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)



print "Decision tree algorithm time after tuning:", round(time()-t0, 3), "s"

accuracy = accuracy_score(labels_test,pred)

print "accuracy after tuning ", accuracy



from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print 'precision = ', precision_score(labels_test,pred)
print 'recall = ', recall_score(labels_test,pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
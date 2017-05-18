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


#########################################################


from sklearn import tree
def decisionTreeSampler(min_samples_split, features_train, features_test, labels_train, labels_test):
    t0 = time()
    clf = tree.DecisionTreeClassifier(min_samples_split= min_samples_split)
    clf = clf.fit(features_train, labels_train)
    print("training time:", round(time()-t0, 3), "s")

    t0 = time()
    pred = clf.predict(features_test)
    print("prediction time:", round(time()-t0, 3), "s")

    from sklearn.metrics import accuracy_score
    print("Prediction accuracy: ", accuracy_score(labels_test, pred))
    print("Chris's emails: ", numpy.sum(pred == 1))

min_samples_split = 2
decisionTreeSampler(min_samples_split, features_train, features_test , labels_train, labels_test)
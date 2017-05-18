#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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
from sklearn import svm
def svmModeler(C, kernelType, features_train, features_test, labels_train, labels_test):
    clf = svm.SVC(kernel= kernelType, C= C)
    t0 = time()
    clf.fit(features_train, labels_train ) 
    print("training time:", round(time()-t0, 3), "s")

    t0 = time()
    pred = clf.predict(features_test)
    print(pred)
    print("prediction time:", round(time()-t0, 3), "s")

    from sklearn.metrics import accuracy_score
    print("Prediction accuracy: ", accuracy_score(labels_test, pred))

C = 6000.0  # SVM regularization parameter
svmModeler(C, "rbf", features_train, features_test , labels_train, labels_test)

print("Chris's emails: ", numpy.sum(pred == 1))
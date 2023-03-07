#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats as st
import scipy.linalg as linalg

true_labels = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw06_true_labels.csv" ,delimiter = ",")

predicted_probabilities = np.genfromtxt("/Users/alitaylanakyurek/Downloads/hw06_predicted_probabilities.csv", delimiter = ",")

tl = true_labels
pp = predicted_probabilities


def roc_curve(true_labels, predicted_probabilities):
    
    thresholds = np.sort(predicted_probabilities)
    thresholds = list(reversed(thresholds))
    fpr = []
    tpr = []
    for threshold in thresholds:
        true_positive = 0
        false_positive = 0
        total_positive = 0
        total_negative = 0
        for i in range(len(true_labels)):
            if true_labels[i] == 1:
                total_positive += 1
                if predicted_probabilities[i] >= threshold:
                    true_positive += 1
            else:
                total_negative += 1
                if predicted_probabilities[i] >= threshold:
                    false_positive += 1
        true_positive_rate = true_positive / total_positive
        false_positive_rate = false_positive / total_negative
        fpr.append(false_positive_rate)
        tpr.append(true_positive_rate)
        
    return fpr, tpr


def integral_with_trapezoid(x, y):
    auc = 0
    for i in range(len(y)-1):
        auc += abs((x[i+1]-x[i])*(y[i+1]+y[i]))/2
    return auc



fpr, tpr = roc_curve(true_labels, predicted_probabilities)
plt.plot(fpr, tpr)
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('TP Rate')
plt.xlabel('FP Rate')
plt.show()

print("The area under the ROC curve is ", integral_with_trapezoid(fpr, tpr))

def pr_curve(true_labels, predicted_probabilities):
    
    thresholds = np.sort(predicted_probabilities)
    thresholds = list(reversed(thresholds))

    tp = 0
    fp = 0
    fn = 0
    precision = []
    recall = []
    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(true_labels)):
            if predicted_probabilities[i] >= threshold:
                if true_labels[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if true_labels[i] == 1:
                    fn += 1
        precision.append(tp/(tp+fp))
        recall.append(tp/(tp+fn))
        
    return precision, recall

precision, recall = pr_curve(true_labels, predicted_probabilities)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()

print("The area under the PR curve is ", integral_with_trapezoid(recall, precision))

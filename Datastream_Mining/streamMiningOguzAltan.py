# -*- coding: utf-8 -*-
"""
@author: Oguz Altan
@Date: 24.05.20
@Title: GE461 Stream Mining Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift
from skmultiflow.trees import HoeffdingTree
from skmultiflow.bayes import NaiveBayes
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer

#Creating 10,000 instances using Random RBF Generator
stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=2, n_features = 10, n_centroids=50)
stream.prepare_for_use()

features_tuple,class_tuple = stream.next_sample(10000)
RBF_Dataset = pd.DataFrame(np.hstack((features_tuple,np.array([class_tuple]).T)))
RBF_Dataset.to_csv('RBF Dataset.csv')

#Creating 10,000 instances using Random RBF Generator Drift with Drift Speed  = 10
stream10 = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state = 50, n_classes = 2, n_features = 10, n_centroids = 50, change_speed = 10, num_drift_centroids=50)
stream10.prepare_for_use()

features_tuple10,class_tuple10 = stream10.next_sample(10000)
RBF_Dataset10 = pd.DataFrame(np.hstack((features_tuple10,np.array([class_tuple10]).T)))
RBF_Dataset10.to_csv('RBF Dataset 10.csv')

features_tuple10,class_tuple10 = stream10.next_sample(10000)
RBF_Dataset10 = pd.DataFrame(np.hstack((features_tuple10,np.array([class_tuple10]).T)))
RBF_Dataset10.to_csv('RBF Dataset 10.csv')

#Creating 10,000 instances using Random RBF Generator Drift with Drift Speed  = 70
stream70 = RandomRBFGeneratorDrift(model_random_state=99, sample_random_state = 50, n_classes = 2, n_features = 10, n_centroids = 50, change_speed = 70, num_drift_centroids=50)
stream70.prepare_for_use()

features_tuple70,class_tuple70 = stream70.next_sample(10000)
RBF_Dataset70 = pd.DataFrame(np.hstack((features_tuple70,np.array([class_tuple70]).T)))
RBF_Dataset70.to_csv('RBF Dataset 70.csv')

train_features, test_features, train_labels, test_labels = train_test_split(RBF_Dataset.iloc[:, 0:10], 
                                                                            RBF_Dataset.iloc[:, 10], 
                                                                            test_size=0.5, random_state=42)
train_features10, test_features10, train_labels10, test_labels10 = train_test_split(RBF_Dataset10.iloc[:, 0:10], 
                                                                            RBF_Dataset10.iloc[:, 10], 
                                                                            test_size=0.5, random_state=42)
train_features70, test_features70, train_labels70, test_labels70 = train_test_split(RBF_Dataset70.iloc[:, 0:10], 
                                                                            RBF_Dataset70.iloc[:, 10], 
                                                                            test_size=0.5, random_state=42)

#Hoeffding Tree with Online Classification
HT = HoeffdingTree()

start = timer() 

correctness_dist = []
for i in range(10000):
   x, y = stream.next_sample()        
   predictHT = HT.predict(x)    
    
   if y == predictHT:                
     correctness_dist.append(1)
   else:
     correctness_dist.append(0)   
   
   HT.partial_fit(x, y.ravel())   
    
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Hoeffding Tree with Online Classification using RBF Dataset: ",accuracy[-1])

correctness_dist = []
for i in range(10000):
    x, y = stream10.next_sample()        
    predictHT10 = HT.predict(x)    
    
    if y == predictHT10:                
      correctness_dist.append(1)
    else:
      correctness_dist.append(0)   
   
    HT.partial_fit(x, y)   
    
time = [i for i in range(1, 10000)]
accuracy10 = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy10)
print("Accuracy of Hoeffding Tree with Online Classification using RBF Dataset 10: ",accuracy10[-1])

correctness_dist = []
for i in range(10000):
    x, y = stream70.next_sample()        
    predictHT70 = HT.predict(x)    
    
    if y == predictHT70:                
      correctness_dist.append(1)
    else:
      correctness_dist.append(0)   
   
    HT.partial_fit(x, y)   
    
time = [i for i in range(1, 10000)]
accuracy70 = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy70)
print("Accuracy of Hoeffding Tree with Online Classification using RBF Dataset 70: ",accuracy70[-1])
end = timer() 
print(end - start)

#Naive Bayes with Online Classification
NB = NaiveBayes()
start = timer() 

correctness_dist = []
for i in range(10000):
   x, y = stream.next_sample()        
   predictNB = NB.predict(x)    
    
   if y == predictNB:                
     correctness_dist.append(1)
   else:
     correctness_dist.append(0)   
   
   NB.partial_fit(x, y)   
    
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Naive Bayes with Online Classification using RBF Dataset: ",accuracy[-1])

correctness_dist = []
for i in range(10000):
    x, y = stream10.next_sample()        
    predictNB10 = NB.predict(x)    
    
    if y == predictNB10:                
      correctness_dist.append(1)
    else:
      correctness_dist.append(0)   
   
    NB.partial_fit(x, y)   
    
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Naive Bayes with Online Classification using RBF Dataset 10: ",accuracy[-1])

correctness_dist = []
for i in range(10000):
    x, y = stream70.next_sample()        
    predictNB70 = NB.predict(x)    
    
    if y == predictNB70:                
      correctness_dist.append(1)
    else:
      correctness_dist.append(0)   
   
    NB.partial_fit(x, y)   
    
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Naive Bayes with Online Classification using RBF Dataset 70: ",accuracy[-1])
end = timer() 
print(end - start)

#Multilayer Perceptron with Online Classification
MLP = MLPClassifier(hidden_layer_sizes = [200,200,200,200],
                            max_iter = 10000,
                            activation = 'tanh',
                            batch_size= 1,
                            solver = 'adam',
                            random_state = 42)
start = timer() 
correctness_dist = []

for i in range(10000):
    
   MLP.partial_fit(x, y,classes = [0,1])   

   x, y = stream.next_sample()        
   predictMLP = MLP.predict(x)    
    
   if y == predictMLP:                
     correctness_dist.append(1)
   else:
     correctness_dist.append(0)   
   
  # MLP.partial_fit(x, y)   
    
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Multilayer Perceptron with Online Classification using RBF Dataset: ",accuracy[-1])

correctness_dist = []

for i in range(10000):
    
    MLP.partial_fit(x, y,classes = [0,1])   

    x, y = stream10.next_sample()        
    predictMLP10 = MLP.predict(x)    
    
    if y == predictMLP10:                
      correctness_dist.append(1)
    else:
      correctness_dist.append(0)   
   
  # MLP.partial_fit(x, y)   
    
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Multilayer Perceptron with Online Classification using RBF Dataset 10: ",accuracy[-1])

correctness_dist = []

for i in range(10000):
    
    MLP.partial_fit(x, y,classes = [0,1])   

    x, y = stream70.next_sample()        
    predictMLP70 = MLP.predict(x)    
    
    if y == predictMLP70:                
      correctness_dist.append(1)
    else:
      correctness_dist.append(0)   
   
  # MLP.partial_fit(x, y)   
    
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Multilayer Perceptron with Online Classification using RBF Dataset 70: ",accuracy[-1])
end = timer() 
print(end - start)
#Ensemble Learning with Majority Voting for Online Classification
correctness_dist = []

for i in range(10000):
    if i < 100:
        X, y = stream.next_sample()
        HT.partial_fit(X, y, classes=[0, 1]) 
        NB.partial_fit(X, y, classes=[0, 1]) 
        MLP.partial_fit(X, y, classes=[0, 1]) 
    else:
        X, y = stream.next_sample()
        prediction = []
        prediction.append(HT.predict(X))
        prediction.append(NB.predict(X))
        prediction.append(MLP.predict(X))
        
        if ((sum(prediction)/2) <1):
            pred = 0
        else: 
            pred = 1
        
        if pred == y:
            correctness_dist.append(1)
        else:
            correctness_dist.append(0) 
        
        HT.partial_fit(X, y) 
        NB.partial_fit(X, y) 
        MLP.partial_fit(X, y)
        
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Majority Voting Online Ensemble Classification using RBF Dataset: ",accuracy[-1])

for i in range(10000):
    if i < 100:
        X, y = stream10.next_sample()
        HT.partial_fit(X, y, classes=[0, 1]) 
        NB.partial_fit(X, y, classes=[0, 1]) 
        MLP.partial_fit(X, y, classes=[0, 1]) 
    else:
        X, y = stream10.next_sample()
        prediction = []
        prediction.append(HT.predict(X))
        prediction.append(NB.predict(X))
        prediction.append(MLP.predict(X))
        
        if ((sum(prediction)/2) <1):
            pred = 0
        else: 
            pred = 1
        
        if pred == y:
            correctness_dist.append(1)
        else:
            correctness_dist.append(0) 
        
        HT.partial_fit(X, y) 
        NB.partial_fit(X, y) 
        MLP.partial_fit(X, y)
        
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Majority Voting Online Ensemble Classification using RBF Dataset 10: ",accuracy[-1])

for i in range(10000):
    if i < 100:
        X, y = stream70.next_sample()
        HT.partial_fit(X, y, classes=[0, 1]) 
        NB.partial_fit(X, y, classes=[0, 1]) 
        MLP.partial_fit(X, y, classes=[0, 1]) 
    else:
        X, y = stream70.next_sample()
        prediction = []
        prediction.append(HT.predict(X))
        prediction.append(NB.predict(X))
        prediction.append(MLP.predict(X))
        
        if ((sum(prediction)/2) <1):
            pred = 0
        else: 
            pred = 1
        
        if pred == y:
            correctness_dist.append(1)
        else:
            correctness_dist.append(0) 
        
        HT.partial_fit(X, y) 
        NB.partial_fit(X, y) 
        MLP.partial_fit(X, y)
        
time = [i for i in range(1, 10000)]
accuracy = [sum(correctness_dist[:i])/len(correctness_dist[:i]) for i in range(1, 10000)]
plt.plot(time, accuracy)
print("Accuracy of Majority Voting Online Ensemble Classification using RBF Dataset 70: ",accuracy[-1])

#Ensemble Learning with Weighted Majority Voting for Batch Classification
pred_weighted = []
yvec = []

HT_W = 0.2
NB_W = 0.3
MLP_W = 0.5

# totac = accuracy_score(test_labels, predHT) + accuracy_score(y_test, predNB) + accuracy_score(y_test, predMLP)

# HT_W = accuracy_score(test_labels, predHT)*1/totacc
# NB_W = accuracy_score(test_labels, predNB)*1/totacc
# MLP_W = accuracy_score(test_labels, predMLP)*1/totacc
        
for i in range(10000):
    if i < 10:
        X, y = stream.next_sample()
        HT.partial_fit(X, y, classes=[0, 1]) 
        NB.partial_fit(X, y, classes=[0, 1]) 
        MLP.partial_fit(X, y, classes=[0, 1]) 
    else:
        X, y = stream.next_sample()       
   
        yvec.append(y)

        prob_0 = (HT_W * HT.predict_proba(X)[0][0]) + (NB_W * NB.predict_proba(X)[0][0]) + (MLP_W * MLP.predict_proba(X)[0][0])
        prob_1 = (HT_W * HT.predict_proba(X)[0][1]) + (NB_W * NB.predict_proba(X)[0][1]) + (MLP_W * MLP.predict_proba(X)[0][1])
        
        if prob_1 > prob_0:
            pred_weighted.append(1)
        else:
            pred_weighted.append(0)
        
        HT.partial_fit(X, y) 
        NB.partial_fit(X, y) 
        MLP.partial_fit(X, y)
                        
accurate = accuracy_score(yvec,pred_weighted)
print("Accuracy of Weighted Majority Voting Online Ensemble Classification using RBF Dataset: ",accurate)

pred_weighted10 = []
yvec10 = []

HT_W10 = 0.2
NB_W10 = 0.3
MLP_W10 = 0.5

# totacc10 = accuracy_score(test_labels10, predHT10) + accuracy_score(y_test10, predNB10) + accuracy_score(y_test10, predMLP10)

# HT_W10 = accuracy_score(test_labels10, predHT10)*1/totacc10
# NB_W10 = accuracy_score(test_labels10, predNB10)*1/totacc10
# MLP_W10= accuracy_score(test_labels10, predMLP10)*1/totacc10
        
for i in range(10000):
    if i < 10:
        X, y = stream.next_sample()
        HT.partial_fit(X, y, classes=[0, 1]) 
        NB.partial_fit(X, y, classes=[0, 1]) 
        MLP.partial_fit(X, y, classes=[0, 1]) 
    else:
        X, y = stream.next_sample()       
   
        yvec10.append(y)

        prob_0 = (HT_W10 * HT.predict_proba(X)[0][0]) + (NB_W10 * NB.predict_proba(X)[0][0]) + (MLP_W10 * MLP.predict_proba(X)[0][0])
        prob_1 = (HT_W10 * HT.predict_proba(X)[0][1]) + (NB_W10 * NB.predict_proba(X)[0][1]) + (MLP_W10 * MLP.predict_proba(X)[0][1])
        
        if prob_1 > prob_0:
            pred_weighted10.append(1)
        else:
            pred_weighted10.append(0)
        
        HT.partial_fit(X, y) 
        NB.partial_fit(X, y) 
        MLP.partial_fit(X, y)
                        
accurate10 = accuracy_score(yvec10,pred_weighted10)
print("Accuracy of Weighted Majority Voting Online Ensemble Classification using RBF Dataset 10: ",accurate10)

pred_weighted70 = []
yvec70 = []

HT_W70 = 0.2
NB_W70 = 0.3
MLP_W70 = 0.5

# totacc70 = accuracy_score(test_labels70, predHT70) + accuracy_score(y_test70, predNB70) + accuracy_score(y_test70, predMLP70)

# HT_W70 = accuracy_score(test_labels70, predHT70)*1/totacc70
# NB_W70 = accuracy_score(test_labels70, predNB70)*1/totacc70
# MLP_W70= accuracy_score(test_labels70, predMLP70)*1/totacc70
        
for i in range(10000):
    if i < 10:
        X, y = stream.next_sample()
        HT.partial_fit(X, y, classes=[0, 1]) 
        NB.partial_fit(X, y, classes=[0, 1]) 
        MLP.partial_fit(X, y, classes=[0, 1]) 
    else:
        X, y = stream.next_sample()       
   
        yvec70.append(y)

        prob_0 = (HT_W70 * HT.predict_proba(X)[0][0]) + (NB_W70 * NB.predict_proba(X)[0][0]) + (MLP_W70 * MLP.predict_proba(X)[0][0])
        prob_1 = (HT_W70 * HT.predict_proba(X)[0][1]) + (NB_W70 * NB.predict_proba(X)[0][1]) + (MLP_W70 * MLP.predict_proba(X)[0][1])
        
        if prob_1 > prob_0:
            pred_weighted70.append(1)
        else:
            pred_weighted70.append(0)
        
        HT.partial_fit(X, y) 
        NB.partial_fit(X, y) 
        MLP.partial_fit(X, y)
                        
accurate70 = accuracy_score(yvec70,pred_weighted70)
print("Accuracy of Weighted Majority Voting Online Ensemble Classification using RBF Dataset 70: ",accurate70)

#Hoeffding Tree Batch Classification
HT = HoeffdingTree()

data = pd.read_csv("RBF Dataset.csv")
data_np = data.to_numpy()
x = data_np[:,:10]
y = data_np[:,-1:]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5, random_state=42)                  
                                                                       
HT.fit(x_train, y_train.ravel())
predHT = HT.predict(x_test)
print("Accuracy score for Hoeffding Tree Batch Classification for RBF Dataset: ",accuracy_score(y_test, predHT))

data = pd.read_csv("RBF Dataset 10.csv")
data_np = data.to_numpy()
x10 = data_np[:,:10]
y10 = data_np[:,-1:]

x_train10,x_test10,y_train10,y_test10 = train_test_split(x10,y10,test_size=0.5, random_state=42)                  
                                                                       
HT.fit(x_train10, y_train10.ravel())
predHT10 = HT.predict(x_test10)
print("Accuracy score for Hoeffding Tree Batch Classification for RBF Dataset 10: ",accuracy_score(y_test10, predHT10))

data = pd.read_csv("RBF Dataset 70.csv")
data_np = data.to_numpy()
x70 = data_np[:,:10]
y70 = data_np[:,-1:]

x_train70,x_test70,y_train70,y_test70 = train_test_split(x70,y70,test_size=0.5, random_state=42)                  
                                                                       
HT.fit(x_train70, y_train70.ravel())
predHT70 = HT.predict(x_test70)
print("Accuracy score for Hoeffding Tree Batch Classification for RBF Dataset 70: ",accuracy_score(y_test70, predHT70))

#Naive Bayes Batch Classification
NB = GaussianNB()

predNB = NB.fit(train_features, train_labels).predict(test_features)
print("Accuracy score for Naive Bayes Batch Classification for RBF Dataset: ",accuracy_score(test_labels, predNB))

predNB10 = NB.fit(train_features10, train_labels10).predict(test_features10)
print("Accuracy score for Naive Bayes Batch Classification for RBF Dataset 10: ",accuracy_score(test_labels10, predNB10))

predNB70 = NB.fit(train_features70, train_labels70).predict(test_features70)
print("Accuracy score for Naive Bayes Batch Classification for RBF Dataset 70: ",accuracy_score(train_labels70, predNB70))

#Multilayer Perceptron with Batch Learning
MLP = MLPClassifier(hidden_layer_sizes = [200,200,200,200],
                              max_iter = 10000,
                              activation = 'tanh',
                              batch_size= 100,
                              solver = 'adam', random_state = 42)


MLP.fit(train_features, train_labels.ravel())
predMLP = MLP.predict(test_features)
print("Accuracy score for Multilayer Perceptron with Batch Classification for RBF Dataset: ",accuracy_score(test_labels, predMLP))

MLP.fit(train_features10, train_labels10.ravel())
predMLP10 = MLP.predict(test_features10)
print("Accuracy score for Multilayer Perceptron with Batch Classification for RBF Dataset 10: ",accuracy_score(test_labels10, predMLP10))

MLP.fit(train_features70, train_labels70.ravel())
predMLP70 = MLP.predict(test_features70)
print("Accuracy score for Multilayer Perceptron with Batch Classification for RBF Dataset 70: ",accuracy_score(test_labels70, predMLP70))

#Ensemble Learning with Majority Voting for Batch Classification
correctness = []

for i in range(5000):      
   if ((predHT[i] + predNB[i] + predMLP[i]) < 2):
       predicted = 0
   else: 
       predicted = 1
        
   if predicted == y_test[i]:
      correctness.append(1)
   else:
      correctness.append(0) 
   
accuracy = [sum(correctness[:i])/len(correctness[:i]) for i in range(1,5000)]
print("Accuracy of Majority Voting Batch Ensemble Classification using RBF Dataset: ",accuracy[-1])

correctness10 = []

for i in range(5000):      
    if ((predHT10[i] + predNB10[i] +predMLP10[i]) < 2):
        predicted10 = 0
    else: 
        predicted10 = 1
        
    if predicted10 == y_test[i]:
      correctness10.append(1)
    else:
      correctness10.append(0) 
   
accuracy10 = [sum(correctness10[:i])/len(correctness10[:i]) for i in range(1,5000)]
print("Accuracy of Majority Voting Batch Ensemble Classification using RBF Dataset 10: ",accuracy10[-1])

correctness70 = []

for i in range(5000):      
    if ((predHT70[i] + predNB70[i] +predMLP70[i]) < 2):
        predicted70 = 0
    else: 
        predicted70 = 1
        
    if predicted70 == y_test[i]:
      correctness70.append(1)
    else:
      correctness70.append(0) 
   
accuracy70 = [sum(correctness70[:i])/len(correctness70[:i]) for i in range(1,5000)]
print("Accuracy of Majority Voting Batch Ensemble Classification using RBF Dataset 70: ",accuracy70[-1])

#Ensemble Learning with Weighted Majority Voting for Batch Classification
prediction_W1 = HT.predict_proba(x_test)
prediction_W2 = NB.predict_proba(x_test)
prediction_W3 = MLP.predict_proba(x_test)

totacc = accuracy_score(y_test, predHT) + accuracy_score(y_test, predNB) + accuracy_score(y_test, predMLP)

HT_W = accuracy_score(y_test, predHT)*1/totacc
NB_W = accuracy_score(y_test, predNB)*1/totacc
MLP_W = accuracy_score(y_test, predMLP)*1/totacc

pred_weighted = []
for i in range (len(x_test)):
    prob_0 = (HT_W * prediction_W1[i][0]) + (NB_W * prediction_W2[i][0]) + (MLP_W * prediction_W3[i][0])
    prob_1 = (HT_W * prediction_W1[i][1]) + (NB_W * prediction_W2[i][1]) + (MLP_W * prediction_W3[i][1])
    if prob_1 > prob_0:
        pred_weighted.append(1)
    else:
        pred_weighted.append(0)
        
accurate = accuracy_score(y_test,pred_weighted)
print("Accuracy of Weighted Majority Voting Batch Ensemble Classification using RBF Dataset: ",accurate)

prediction_W1 = HT.predict_proba(x_test10)
prediction_W2 = NB.predict_proba(x_test10)
prediction_W3 = MLP.predict_proba(x_test10)

totacc10 = accuracy_score(y_test10, predHT10) + accuracy_score(y_test10, predNB10) + accuracy_score(y_test10, predMLP10)

HT_W10 = accuracy_score(y_test10, predHT10)*1/totacc10
NB_W10 = accuracy_score(y_test10, predNB10)*1/totacc10
MLP_W10 = accuracy_score(y_test10, predMLP10)*1/totacc10

pred_weighted10 = []
for i in range (len(x_test)):
    prob_0 = (HT_W10 * prediction_W1[i][0]) + (NB_W10 * prediction_W2[i][0]) + (MLP_W10 * prediction_W3[i][0])
    prob_1 = (HT_W10 * prediction_W1[i][1]) + (NB_W10 * prediction_W2[i][1]) + (MLP_W10 * prediction_W3[i][1])
    if prob_1 > prob_0:
        pred_weighted10.append(1)
    else:
        pred_weighted10.append(0)
        
accurate10 = accuracy_score(y_test10,pred_weighted10)
print("Accuracy of Weighted Majority Voting Batch Ensemble Classification using RBF Dataset 10: ",accurate10)

prediction_W1 = HT.predict_proba(x_test70)
prediction_W2 = NB.predict_proba(x_test70)
prediction_W3 = MLP.predict_proba(x_test70)

totacc70 = accuracy_score(y_test70, predHT70) + accuracy_score(y_test70, predNB70) + accuracy_score(y_test70, predMLP70)

HT_W70 = accuracy_score(y_test70, predHT70)*1/totacc70
NB_W70 = accuracy_score(y_test70, predNB70)*1/totacc70
MLP_W70 = accuracy_score(y_test70, predMLP70)*1/totacc70

pred_weighted70 = []
for i in range (len(x_test)):
    prob_0 = (HT_W70 * prediction_W1[i][0]) + (NB_W70 * prediction_W2[i][0]) + (MLP_W70 * prediction_W3[i][0])
    prob_1 = (HT_W70 * prediction_W1[i][1]) + (NB_W70 * prediction_W2[i][1]) + (MLP_W70 * prediction_W3[i][1])
    if prob_1 > prob_0:
        pred_weighted70.append(1)
    else:
        pred_weighted70.append(0)
        
accurate70 = accuracy_score(y_test70,pred_weighted70)
print("Accuracy of Weighted Majority Voting Batch Ensemble Classification using RBF Dataset 70: ",accurate70)

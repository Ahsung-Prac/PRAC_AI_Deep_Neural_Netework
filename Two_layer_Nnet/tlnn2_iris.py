#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.pardir)
sys.path.append("../../")
from sklearn.datasets import load_iris
from ai_func import *
from tlnn2 import TwoLayerNeuralNetwork2


# In[2]:

"""
Load Data and one-hot encoding
"""
iris = load_iris()
X = iris.data
Y = iris.target
Y = one_hot(Y)
y_name = iris.target_names
print("feature 수: ",X.shape[1])
print("class :",y_name)


# In[88]:


"""
Train data, Test data Setting
"""
# 인덱스 suffle
suffle_mask = np.random.choice(X.shape[0],X.shape[0],replace=False)

# data 80퍼센트는 Train용, 20퍼센트는 Test용
bound = int(0.8 * X.shape[0])

x_train = X[suffle_mask[:bound]]
x_test = X[suffle_mask[bound:]]
y_train = Y[suffle_mask[:bound]]
y_test = Y[suffle_mask[bound:]]


# In[156]:


"""
init_model
"""
input_size = X.shape[1]
hidden_size = 10
output_size = Y.shape[1]

tn2 = TwoLayerNeuralNetwork2(input_size,hidden_size,output_size)


# In[161]:


"""
learn
"""
epochs = 120
batch_size = 20
lr = 0.03
# Do not use Test_data, just record test_accuracy every epoch
tn2.learn(x_train,y_train,test_X = x_test,test_Y = y_test, epochs = epochs, batch_size = batch_size,lr= lr,hist=True)


# In[162]:

"""
Test Case accuracy
"""
# Test Case 정확도
print("-----Test case 정확도-----")
print("target :",tn2.accuracy(x_test,y_test))


# In[163]:

"""
Accuracy Graph
"""
print("hidden size: ",hidden_size)
print("epochs: ",epochs)
print("batch size: ",batch_size)
x = np.arange(len(tn2.train_accuracylist))
plt.plot(x, tn2.train_accuracylist, label='train accuracy')
plt.plot(x, tn2.test_accuracylist, label='train accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.show()


# In[164]:

"""
loss Graph
"""
print("hidden size: ",hidden_size)
print("epochs: ",epochs)
print("batch size: ",batch_size)

maxloss = max(tn2.train_losslist)
x = np.arange(len(tn2.train_losslist))
plt.plot(x, tn2.train_losslist, label='train loss' ,linestyle='--')
plt.plot(x, tn2.test_losslist, label='test loss' ,linestyle='--')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, maxloss)
plt.legend(loc='lower right')
plt.show()


# In[ ]:





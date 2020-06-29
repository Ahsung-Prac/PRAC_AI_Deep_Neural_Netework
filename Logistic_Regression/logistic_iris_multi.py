#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.pardir)
sys.path.append("../../")
from sklearn.datasets import load_iris
from logisticReg import Lreg


# In[2]:


iris = load_iris()
X = iris.data
iris_Y = iris.target

Y = iris_Y.copy()

# 원 핫 인코딩
num = np.unique(Y,axis=0)
num = num.shape[0]
Y = np.eye(num)[Y].astype(np.int)

y_name = iris.target_names
print("feature 수: ",X.shape[1])


# In[118]:


# 인덱스 suffle
suffle_mask = np.random.choice(X.shape[0],X.shape[0],replace=False)

# data 80퍼센트는 Train용, 20퍼센트는 Test용
bound = int(0.8 * X.shape[0])

x_train = X[suffle_mask[:bound]]
x_test = X[suffle_mask[bound:]]
y_train = Y[suffle_mask[:bound]]
y_test = Y[suffle_mask[bound:]]


# In[119]:


# feature * class 수 = 4*3 weight layer + bias 
reg = Lreg(X.shape[1],y_name.shape[0],"m")


# In[120]:


reg.reset_weight()
reg.learn(x_train,y_train,epoch=200,batch_size=120,lr=0.07)


# In[121]:


# Test Case 정확도
print("-----Test case 정확도-----")
print("target :",reg.accuracy(x_test,y_test))


# In[122]:


x = np.arange(len(reg.train_accuracylist))
plt.plot(x, reg.train_accuracylist, label='train accuracy')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.show()


# In[123]:


x = np.arange(len(reg.train_costlist))
cost = np.array(reg.train_costlist).T
maxcost = cost.max()
plt.plot(x, cost[0], label='train0 cost' ,linestyle='--')
plt.plot(x, cost[1], label='train1 cost',linestyle='--')
plt.plot(x, cost[2], label='train2 cost',linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, maxcost + 0.1)
plt.legend(loc='lower right')
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pylab as plt
import numpy as np
from sklearn.datasets import load_iris
from knn import KNN


# In[2]:


# sepal length,width  petal length width
# 꽃 받침, 꽃잎

# Three class
# 0: Setosa
# 1: Versicolor
# 2 Virginica

iris = load_iris()
X = iris.data
Y = iris.target
y_name = iris.target_names
print(y_name)


# In[3]:


# Sepal min,max length
x1_min, x1_max = X[:,0].min() -0.5 , X[:,0].max() + 0.5
# Sepal min,max width
x2_min, x2_max = X[:,1].min() - 0.5 , X[:,1].max() + 0.5

plt.figure(2,figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=Y,cmap = plt.cm.Set1,edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.xticks(())
plt.yticks(())
plt.show()


# In[4]:


for_test = np.array([ ((i%15) == 14) for i in range(Y.shape[0])])
for_train = ~for_test
x_train = X[for_train]
y_train = Y[for_train]
x_test = X[for_test]
y_test = Y[for_test]


# In[5]:


K_list = [3,5,10]
knn_iris = KNN(1,x_train,y_train,y_name)

# weight 다시 셋팅 예제
#knn_iris.set_weight_func(lambda distance: (1 / np.log(distance+2)))

for k in K_list:
    knn_iris.set_k(k)
    print("Test k = ",k)
    knn_iris.show_dim()
    print()
    for i in range(y_test.shape[0]):
        knn_iris.get_nearest_k(x_test[i])
        print("Test Data Index:",i,", Weight Computued class:",knn_iris.weighted_majority_vote(),",True class:",y_name[y_test[i]])
        knn_iris.obtain_majority_vote()
        print("Nearest Count [setosa,versicolor,virginica] = ",knn_iris.obtain_vote)
        print("Weight Sum(int) [setosa,versicolor,virginica] = ",knn_iris.weight_vote.astype('uint32'),"\n")
    print()


# In[ ]:





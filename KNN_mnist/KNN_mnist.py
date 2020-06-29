#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys, os
sys.path.append(os.pardir)
# 부모 디렉토리에서 import할 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
# mnist data load할 수 있는 함수 import
from PIL import Image
from knn import KNN
import numpy as np


# In[8]:


# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize(정규화): 0~1 실수로. 그렇지 않으면 0~255
# 정구화한 값으로 get
(x_train, t_train), (x_test, t_test) =load_mnist(flatten=True, normalize=True)
label_name = np.array(['0','1','2','3','4','5','6','7','8','9'])
print("train data number: ",t_train.shape[0])
print("test data number: ",t_test.shape[0])


# In[9]:


knn_mnist = KNN(3,x_train,t_train,label_name)
knn_mnist.show_dim()


# In[10]:


knn_mnist.set_weight_func( lambda d : 1/(d+0.001) )

cnt = 0
test_Num = 100
tp = int(test_Num/10)
sample = np.random.randint(0,t_test.shape[0],test_Num)

for i in range(test_Num):
    knn_mnist.get_nearest_k(x_test[sample[i]])
    pre =  knn_mnist.weighted_majority_vote()
    t = label_name[t_test[sample[i]]]
    if pre == t: cnt +=1
    if (i % tp) == (tp-1):    
        print(i+1,"번째  predict: ", pre,"  True class: ",t, "accuracy:{}%".format(cnt/(i+1)*100))
print()
print("결과 정확도: {}%".format(cnt/test_Num*100))


# In[11]:


# 각 이미지의 행의 총합과 열의 총합을 concatenate한 56개의 hand-crafted feature 
tmp_train_x1 = x_train.reshape(x_train.shape[0],28,28).sum(2)
tmp_train_x2 = x_train.reshape(x_train.shape[0],28,28).sum(1)

tmp_test_x1 = x_test.reshape(x_test.shape[0],28,28).sum(2)
tmp_test_x2 = x_test.reshape(x_test.shape[0],28,28).sum(1)

x_train_hand = np.append(tmp_train_x1,tmp_train_x2,axis = 1)
x_test_hand = np.append(tmp_test_x1,tmp_test_x2,axis = 1)
print(x_train_hand.shape)
print(x_test_hand.shape)


# In[12]:


knn_mnist_hand = KNN(3,x_train_hand,t_train,label_name)
knn_mnist_hand.show_dim()


# In[13]:


knn_mnist_hand.set_weight_func( lambda d : 1/(d+0.01) )

cnt = 0
test_Num = 1000
tp = int(test_Num/10)
sample = np.random.randint(0,t_test.shape[0],test_Num)

for i in range(test_Num):
    knn_mnist_hand.get_nearest_k(x_test_hand[sample[i]])
    pre =  knn_mnist_hand.weighted_majority_vote()
    t = label_name[t_test[sample[i]]]
    if pre == t: cnt +=1
    if (i % tp) == (tp-1):    
        print(i+1,"번째  predict: ", pre,"  True class: ",t, "accuracy:{}%".format(cnt/(i+1)*100))
print()
print("결과 정확도: {}%".format(cnt/test_Num*100))


# In[14]:


# 28*28 이미지를 2*2 정사각형 196개로 나누어
# 각 정사각형의 총합을 feature로하는 hand-craft feature

tmp_train_x = x_train.reshape(x_train.shape[0],14,2,28).sum(2)
tmp2_train_x = tmp_train_x.reshape(tmp_train_x.shape[0],14,14,2).sum(3)
x_train_hand2 = tmp2_train_x.reshape(tmp_train_x.shape[0],-1)

tmp_test_x = x_test.reshape(x_test.shape[0],14,2,28).sum(2)
tmp2_test_x = tmp_test_x.reshape(tmp_test_x.shape[0],14,14,2).sum(3)
x_test_hand2 = tmp2_test_x.reshape(tmp_test_x.shape[0],-1)

print(x_train_hand2.shape)
print(x_test_hand2.shape)


# In[15]:


knn_mnist_hand2 = KNN(3,x_train_hand2,t_train,label_name)
knn_mnist_hand2.show_dim()


# In[16]:


knn_mnist_hand2.set_weight_func( lambda d : 1/(d+0.01) )

cnt = 0
test_Num = 1000
tp = int(test_Num/10)
sample = np.random.randint(0,t_test.shape[0],test_Num)

for i in range(test_Num):
    knn_mnist_hand2.get_nearest_k(x_test_hand2[sample[i]])
    pre =  knn_mnist_hand2.weighted_majority_vote()
    t = label_name[t_test[sample[i]]]
    if pre == t: cnt +=1
    if (i % tp) == (tp-1):    
        print(i+1,"번째  predict: ", pre,"  True class: ",t, "accuracy:{}%".format(cnt/(i+1)*100))
print()
print("결과 정확도: {}%".format(cnt/test_Num*100))


# In[17]:


# 28*28 이미지를 4*4 정사각형 49개로 나누어
# 각 정사각형의 총합을 feature로하는 hand-craft feature

tmp_train_x = x_train.reshape(x_train.shape[0],7,4,28).sum(2)
tmp2_train_x = tmp_train_x.reshape(tmp_train_x.shape[0],7,7,4).sum(3)
x_train_hand3 = tmp2_train_x.reshape(tmp_train_x.shape[0],-1)

tmp_test_x = x_test.reshape(x_test.shape[0],7,4,28).sum(2)
tmp2_test_x = tmp_test_x.reshape(tmp_test_x.shape[0],7,7,4).sum(3)
x_test_hand3 = tmp2_test_x.reshape(tmp_test_x.shape[0],-1)

print(x_train_hand3.shape)
print(x_test_hand3.shape)


# In[18]:


knn_mnist_hand3 = KNN(3,x_train_hand3,t_train,label_name)
knn_mnist_hand3.show_dim()


# In[19]:


knn_mnist_hand3.set_weight_func( lambda d : 1/(d+0.01) )

cnt = 0
test_Num = 10000
tp = int(test_Num/10)
sample = np.random.randint(0,t_test.shape[0],test_Num)

for i in range(test_Num):
    knn_mnist_hand3.get_nearest_k(x_test_hand3[sample[i]]) 
    pre =  knn_mnist_hand3.weighted_majority_vote()
    t = label_name[t_test[sample[i]]]
    if pre == t: cnt +=1
    if (i % tp) == (tp-1):    
        print(i+1,"번째  predict: ", pre,"  True class: ",t, "accuracy:{}%".format(cnt/(i+1)*100))
print()
print("결과 정확도: {}%".format(cnt/test_Num*100))


# In[ ]:





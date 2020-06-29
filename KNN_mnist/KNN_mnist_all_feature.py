#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys, os
sys.path.append(os.pardir)
sys.path.append("../../")
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

print("\nALL_Feature KNN MNIST\n")
print("train data number: ",t_train.shape[0])
print("test data number: ",t_test.shape[0])


# In[9]:

# Input value is All feature !!
knn_mnist = KNN(3,x_train,t_train,label_name)
knn_mnist.show_dim()


# In[10]:

print("\nPredict Start\n")

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





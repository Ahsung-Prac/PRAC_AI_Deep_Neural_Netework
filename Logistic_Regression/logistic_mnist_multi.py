#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.pardir)
sys.path.append("../../")
from dataset.mnist import load_mnist
from logisticReg import Lreg
from logisticReg import one_hot


# In[2]:


# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize(정규화): 0~1 실수로. 그렇지 않으면 0~255
# 정구화한 값으로 get
(x_train, t_train_f), (x_test, t_test_f) =load_mnist(flatten=True, normalize=True)

#라벨 이름 셋팅.
label_name = np.array(['0','1','2','3','4','5','6','7','8','9'])

print("train data number: ",t_train_f.shape[0])
print("test data number: ",t_test_f.shape[0])


# In[3]:


# 원 핫 인코딩
t_train = one_hot(t_train_f)
t_test = one_hot(t_test_f)


# In[4]:


# 모델 생성
# feature * class 수 = 784*10 weight layer +  bias 
reg = Lreg(x_train.shape[1],label_name.shape[0],"m")


# In[33]:


reg.reset_weight()

# 6만개 데이터 학습시작
reg.learn(x_train,t_train,epoch=20,batch_size=1000,evalu_mode='s',lr = 0.05)


# In[34]:


# Test Case 정확도
print("-----Test case 정확도-----")
print("Accuracy :",reg.accuracy(x_test,t_test))


# In[35]:


# Test case 평가!
reg.evaluate(x_test,t_test,ev_target="test")


# In[38]:


x = np.arange(len(reg.train_accuracylist))
plt.plot(x, reg.train_accuracylist, label='train accuracy')
plt.xlabel("steps")
plt.ylabel("accuracy")
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.show()


# In[39]:


x = np.arange(len(reg.train_costlist))
cost = np.array(reg.train_costlist).T
maxcost = cost.max()
for n in range(10):
    plt.plot(x, cost[n], label='train'+str(n)+'cost',linestyle='--')
plt.xlabel("steps")
plt.ylabel("accuracy")
plt.ylim(0, maxcost + 0.1)
plt.legend(loc='lower right')
plt.show()


# In[ ]:





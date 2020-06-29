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
from logisticReg import binaryTarget
from logisticReg import step_function


# In[2]:


# training data, test data
# flatten: 이미지를 1차원 배열로 읽음
# normalize(정규화): 0~1 실수로. 그렇지 않으면 0~255
# 정구화한 값으로 get
(x_train, t_train), (x_test, t_test) =load_mnist(flatten=True, normalize=True)

#라벨 이름 셋팅.
label_name = np.array(['0','1','2','3','4','5','6','7','8','9'])

print("train data number: ",t_train.shape[0])
print("test data number: ",t_test.shape[0])


# In[3]:


t0_train = binaryTarget(t_train,target = 0)
t1_train = binaryTarget(t_train,target = 1)
t2_train = binaryTarget(t_train,target = 2)
t3_train = binaryTarget(t_train,target = 3)
t4_train = binaryTarget(t_train,target = 4)
t5_train = binaryTarget(t_train,target = 5)
t6_train = binaryTarget(t_train,target = 6)
t7_train = binaryTarget(t_train,target = 7)
t8_train = binaryTarget(t_train,target = 8)
t9_train = binaryTarget(t_train,target = 9)

t0_test = binaryTarget(t_test,target = 0)
t1_test = binaryTarget(t_test,target = 1)
t2_test = binaryTarget(t_test,target = 2)
t3_test = binaryTarget(t_test,target = 3)
t4_test = binaryTarget(t_test,target = 4)
t5_test = binaryTarget(t_test,target = 5)
t6_test = binaryTarget(t_test,target = 6)
t7_test = binaryTarget(t_test,target = 7)
t8_test = binaryTarget(t_test,target = 8)
t9_test = binaryTarget(t_test,target = 9)


# In[4]:


reg0 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg1 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg2 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg3 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg4 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg5 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg6 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg7 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg8 = Lreg(x_train.shape[1],label_name.shape[0],"b")
reg9 = Lreg(x_train.shape[1],label_name.shape[0],"b")


# In[5]:


# 데이터 6만개 너무 많다.. 몇개만쓰자..
num = 10000
shuffle_mask = np.random.choice(x_train.shape[0],num,replace=False)
x_train_s = x_train[shuffle_mask]
t0_train_s = t0_train[shuffle_mask]
t1_train_s = t1_train[shuffle_mask]
t2_train_s = t2_train[shuffle_mask]
t3_train_s = t3_train[shuffle_mask]
t4_train_s = t4_train[shuffle_mask]
t5_train_s = t5_train[shuffle_mask]
t6_train_s = t6_train[shuffle_mask]
t7_train_s = t7_train[shuffle_mask]
t8_train_s = t8_train[shuffle_mask]
t9_train_s = t9_train[shuffle_mask]


# In[6]:


# 학습모델 weight reset
reg0.reset_weight()
reg1.reset_weight()
reg2.reset_weight()
reg3.reset_weight()
reg4.reset_weight()
reg5.reset_weight()
reg6.reset_weight()
reg7.reset_weight()
reg8.reset_weight()
reg9.reset_weight()
# 6만개 데이터 학습시작
reg0.learn(x_train_s,t0_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg1.learn(x_train_s,t1_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg2.learn(x_train_s,t2_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg3.learn(x_train_s,t3_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg4.learn(x_train_s,t4_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg5.learn(x_train_s,t5_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg6.learn(x_train_s,t6_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg7.learn(x_train_s,t7_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg8.learn(x_train_s,t8_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)
reg9.learn(x_train_s,t9_train_s,epoch=200,batch_size=num,evalu_mode='e',lr = 0.03)


# In[7]:


# Test Case 정확도
print("-----Test case 정확도-----")
print("target 0:",reg0.accuracy(x_test,t0_test))
print("target 1:",reg9.accuracy(x_test,t1_test))
print("target 2:",reg0.accuracy(x_test,t2_test))
print("target 3:",reg9.accuracy(x_test,t3_test))
print("target 4:",reg0.accuracy(x_test,t4_test))
print("target 5:",reg9.accuracy(x_test,t5_test))
print("target 6:",reg0.accuracy(x_test,t6_test))
print("target 7:",reg9.accuracy(x_test,t7_test))
print("target 8:",reg0.accuracy(x_test,t8_test))
print("target 9:",reg9.accuracy(x_test,t9_test))


# In[8]:


reg9.evaluate(x_test,t9_test)


# In[14]:


x = np.arange(len(reg0.train_accuracylist))
plt.plot(x, reg0.train_accuracylist, label='train0 accuracy', linestyle='--')
plt.plot(x, reg1.train_accuracylist, label='train1 accuracy', linestyle='--')
plt.plot(x, reg2.train_accuracylist, label='train2 accuracy', linestyle='--')
plt.plot(x, reg3.train_accuracylist, label='train3 accuracy', linestyle='--')
plt.plot(x, reg4.train_accuracylist, label='train4 accuracy', linestyle='--')
plt.plot(x, reg5.train_accuracylist, label='train5 accuracy', linestyle='--')
plt.plot(x, reg6.train_accuracylist, label='train6 accuracy', linestyle='--')
plt.plot(x, reg7.train_accuracylist, label='train7 accuracy', linestyle='--')
plt.plot(x, reg8.train_accuracylist, label='train8 accuracy', linestyle='--')
reg9.train_accuracylist.pop()#이유는 모르겠는데, 1개가 더들어가있어서 그림을 못그림 ㅠ
plt.plot(x, reg9.train_accuracylist, label='train9 accuracy', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.show()


# In[17]:


maxlist = [max(reg0.train_costlist),max(reg1.train_costlist),max(reg2.train_costlist),max(reg3.train_costlist),max(reg4.train_costlist),max(reg5.train_costlist),max(reg6.train_costlist),max(reg7.train_costlist),max(reg8.train_costlist),max(reg9.train_costlist)]#,max(reg2.train_costlist)]
maxcost = max(maxlist)

x = np.arange(len(reg0.train_costlist))
plt.xlabel("epochs")
plt.plot(x, reg0.train_costlist, label='train0 cost', linestyle='--')
plt.plot(x, reg1.train_costlist, label='train1 cost', linestyle='--')
plt.plot(x, reg2.train_costlist, label='train2 cost', linestyle='--')
plt.plot(x, reg3.train_costlist, label='train3 cost', linestyle='--')
plt.plot(x, reg4.train_costlist, label='train4 cost', linestyle='--')
plt.plot(x, reg5.train_costlist, label='train5 cost', linestyle='--')
plt.plot(x, reg6.train_costlist, label='train6 cost', linestyle='--')
plt.plot(x, reg7.train_costlist, label='train7 cost', linestyle='--')
plt.plot(x, reg8.train_costlist, label='train8 cost', linestyle='--')
reg9.train_costlist.pop()#이유는 모르겠는데, 1개가 더들어가있어서 그림을 못그림 ㅠ
plt.plot(x, reg9.train_costlist, label='train9 cost', linestyle='--')
plt.ylabel("cost")
plt.ylim(0,maxcost)
plt.legend(loc='lower right')
plt.show()


# In[ ]:





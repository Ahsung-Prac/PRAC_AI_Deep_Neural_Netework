#!/usr/bin/env python
# coding: utf-8

# In[95]:


import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.pardir)
sys.path.append("../../")
from dataset.mnist import load_mnist
from sklearn.datasets import load_iris



def sigmoid(x):
    eMin = -np.log(np.ﬁnfo(type(0.1)).max) 
    zSafe = np.array(np.maximum(x, eMin))
    res = (1.0/(1+np.exp(-zSafe)))
    
    # 가끔 0과 1이 되버리는 오류가 난다길래,,
    #  안전하게 하는 코드를 넣어봤음..
    # sigmoid는 0과 1이 나올 수 없는데, 컴퓨터다보니까 너무 작은수는 숫자가 넘어가버리는듯 하다.
    zero_idx = (res == 0.0)
    one_idx = (res == 1.0)
    res[zero_idx] = 1e-7
    res[one_idx] = 1-(1e-7)
    return res

#one hot encoding
def one_hot(x):
    num = np.unique(x,axis=0)
    num = num.shape[0]
    
    return np.eye(num)[x].astype(np.int)

#binary classfication에서 true와 false의 boundary를 계산할 계단함수
def step_function(x, bound = 0):
    a = np.array(x> bound)
    return a.astype(np.int) 

# binary class또한 multi class와 같은 코드로 학습을 진행 할 수 있도록 차원을 맞추고
# binary cassfication은 True or False이므로, target만 정답으로 처리 나머지는 false
def binaryTarget(y_train,target):
    return np.array(y_train == target).astype(np.int).reshape(-1,1)

# numpy suffle...쓰지 않아도 된다..
def shuffle_np(arr,choice_num=-1,overlap = False):
    if choice_num==-1: 
        choice_num = arr.shape[0]
    
    suffle_mask = np.random.choice(arr.shape[0],choice_num,replace=overlap)
    return arr[suffle_mask]


# Logistic Regression Class
class Lreg():
    
    # feature의 개수
    # 분류할 종류의 개수 ( binary classfication의 경우 영향 없다. )
    # classfication mode 입력
    def __init__(self,feature_num,class_num,flag= "m"):
        self.feature_num = feature_num
        self.class_num = class_num
        
        if(flag == "m"):
            self.mode = "multi"
        else:
            self.mode = "binary"
        
        self.reset_weight() # weight값과 bias 설정
            
            
    # weight값을 모드,feature수,class 수 맞는 차원으로 셋팅
    # weight값은 He 랜덤값으로 만든다.
    def reset_weight(self):
        if(self.mode == "binary"):
            self.w = (np.random.randn(self.feature_num,1) * np.sqrt(2.0 /self.feature_num))
            self.b = np.random.randn(1,1);         
            
        else:
            self.w = (np.random.randn(self.feature_num,self.class_num) * np.sqrt(2.0 /self.feature_num))
            self.b = (np.random.randn(1,self.class_num)* np.sqrt(2.0 /self.feature_num))
    
    
    
    # 입력값 x를 weight numpy와 행렬곱 후 sigmoid를 하여
    # 예측한 값을 린턴한다.
    def predict(self,x):
        if(x.ndim == 1):
            x = x.reshape(1,x.size)
        if(x.shape[1] != self.w.shape[0]):
            print("Dot할 수없음, 입력값_shape:",x.shape,"weight_shape:",self.w.shape)
            
        a = x.dot(self.w) + self.b;
        h = sigmoid(a)
        return h
    
    
    
    # 입력 데이터와, 정답 데이터를 받아 정확도를 계산한다.
    # 모드에 따라 다르게 작동한다.
    def accuracy(self,X,Y):
        if(X.ndim == 1): X = X.reshape(1,X.size)
        if(Y.ndim == 1) :Y = Y.reshape(1,Y.size)
        
        
        # 가장 큰 값과 정답 lable을 비교
        if self.mode == "multi":          
            predict_idx = self.predict(X).argmax(1)

            trcnt = 0
            for i in range(X.shape[0]):
                if Y[i][predict_idx[i]] == 1 : trcnt+=1
        
        # step함수를 통해 true or false로 변환후
        # 정답 레이블과 비교
        else:
            predict = step_function(self.predict(X),0.5)
            
            trcnt = 0
            for i in range(X.shape[0]):
                if Y[i][0] == predict[i][0] : trcnt+=1
        
        # 맞은 수 / 장답
        return trcnt/X.shape[0]
    
    
    
    # 한스텝마다 미분값을 learning_rate 비율만큼 빼주며 wegiht를 갱신한다.
    def step(self,x,y,learning_rate):
        if(x.ndim == 1): x = x.reshape(1,x.size)
        if(y.ndim == 1) :y = y.reshape(1,y.size)
            
        res = self.predict(x)
        xj = x.T.copy() # 전치, 한 행은 모든 데이터의 j번째 feature를 의미
        batch_size = x.shape[0] # batch size는 입력데이터의 개수!
        
        # j번째 feature는 weight의 한 행에 모두 해당하므로
        # weight의 한 행씩 한번에 갱신한다.
        for i in range(self.w.shape[0]):
            self.w[i] = self.w[i] - learning_rate*(((res-y)*xj[i].reshape(-1,1))).sum(0)/batch_size
        
        self.b = self.b - learning_rate*(((res-y))).sum(0)/batch_size
    
    
    
    # 훈련함수
    # batch 수 만큼 한 step의 학습에 사용한다
    # 본 과제에서는 batch는 총 데이터의 개수이나, mnist의 경우 그 수가 6만개로 너무 방대해
    # 여러가지 테스트 과정을 해보기 위해서 batch size를 조정할 수 있도록 코딩해보았다.
    # evaluate_mode에 따라, accuracy와 cost를 step마다 저장할지, epoch마다 저장할지 정한다.
    # 테스트 데이터를 넣게되면, 학습에 사용하지는 않지만, 매 학습마다 테스트 데이터에 대한 평가를 해볼 수 있다.
    def learn(self,X,Y,epoch = 30,batch_size = 20,lr = 0.07,test_X = None,test_Y= None,evalu_mode="e",verb=True):
        if(X.ndim == 1): X = X.reshape(1,X.size)
        if(Y.ndim == 1) :Y = Y.reshape(1,Y.size)
        if(X.shape[0] < batch_size) : 
            batch_size = X.shape[0]
                    
        self.train_accuracylist = []
        self.test_accuracylist = []
        self.train_costlist = []
        self.test_costlist = []

        #학습전에 평가를 해본다.
        print("Before Learning:")
        self.evaluate(X,Y,verb=verb)
        
        # 에폭당 배치 반복 학습
        # 이번 Logistic Regression과제에서는 batchsize = 총 데이터의 수 이기 때문에
        # epochs가 곧 학습을 갱신한 step의 수가 된다.  
        batch_per_epoch = X.shape[0] // batch_size
        if X.shape[0] % batch_size != 0 : batch_per_epoch+=1
        
        for i in range(epoch):
            
            # one epoch Suffle
            # 같은 데이터를 매번 같은 순서로 반복 학습하게 되면 weight가 제대로 학습이 안되므로 suffle
            # 매번 배치사이즈만큼 랜덤값을 뽑으면, 데이터마다 학습 기회가 달라질 수 있으므로, 
            # epoch마다 셔플한 전체 데이터를 batch size만큼 step을 계산한다.
            suffle_mask = np.random.choice(X.shape[0],X.shape[0],replace=False)
            x_train = X[suffle_mask]
            y_train = Y[suffle_mask]
            
            for n in range(batch_per_epoch):  
                start = n*batch_size
                end = min(X.shape[0],(n+1)*batch_size)
                
                self.step(x_train[start:end],y_train[start:end],lr)
                
                if evalu_mode != "e":
                    if verb:
                        print(batch_per_epoch*i + n)
                    self.evaluate(x_train[start:end],y_train[start:end],verb=verb)
                     
                    # test data가 있다면, 훈련에 사용하지는 않지만,
                    # 정확도 추이를 기록한다.
                    if (test_X is not None) and (test_Y is not None) :
                        self.evaluate(x_test,y_test,ev_target="test",verb=verb)
             
            
            # 한 에폭마다 마지막 훈련데이터의 정확도와 cost값을 저장한다.
            if evalu_mode=="e":
                if verb:
                    print(i+1)
                self.evaluate(x_train[start:end],y_train[start:end],verb=verb)
                 # test data가 있다면, 훈련에 사용하지는 않지만,
                 # 정확도 추이를 기록한다.
                if (test_X is not None) and (test_Y is not None) :
                        self.evaluate(test_X,test_Y,ev_target="test",verb=verb)
            
    
    
    # data의 정확도와 Cost값을 계산하여 출력해준다.
    # 각 학습마다 배열에 기록함으로서 나중에 그래프로 보기 용이하게 한다.
    # test 데이터는 매 학습에 참여하는 데이터는 아니지만 평가해볼수는 있다.
    def evaluate(self,x_data,y_data,ev_target = "train",verb = True):
        
            train_accur = self.accuracy(x_data,y_data)
            cost = self.cost(x_data,y_data)
            
            if ev_target == "train":
                target = "| Train_"
                accuracylist = self.train_accuracylist
                costlist = self.train_costlist
            else:
                target = "| Test_"
                accuracylist = self.test_accuracylist
                costlist = self.test_costlist
 
            accuracylist.append(train_accur)
            costlist.append(cost)
            
            if verb:
                print(target+"Accuracy:",train_accur,"| COST:",cost)
        
    
  
    def cost(self,x,y):
        if(x.ndim == 1): x = x.reshape(1,x.size)
        if(y.ndim == 1) :y = y.reshape(1,y.size)
        h = self.predict(x)
        
        return -((y*np.log(h)+(1-y)*np.log(1-h)).sum(0))/x.shape[0]


# In[268]:


np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})





#!/usr/bin/env python
# coding: utf-8

# In[84]:

import numpy as np
import sys, os
sys.path.append(os.pardir)
sys.path.append("../../")
from ai_func import *


# target data는 one-hot encoding
# 모든 np배열을 2차원으로 통일. 1차원 행렬은 1*n 형식의 2차원으로 reshape
class TwoLayerNeuralNetwork2():
    def __init__(self,input_size,hidden_size,output_size):
        # set size
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # initalize parmaters, each Layer(weight, bias)
        self.params = {}
        self.init_params();
        
        # init_history
        self.train_accuracylist = []
        self.test_accuracylist = []
        self.train_losslist = []
        self.test_losslist = []
    
    # weight값은 He 랜덤값으로 만든다.
    def init_params(self):
        #input layer
        self.params['W1'] = np.random.randn(self.input_size,self.hidden_size) * np.sqrt(2.0 / self.hidden_size)
        self.params['b1'] = np.random.randn(1,self.hidden_size)* np.sqrt(2.0 / self.hidden_size)
        
        #hidden layer
        self.params['W2'] = np.random.randn(self.hidden_size,self.output_size)* np.sqrt(2.0 / self.output_size)
        self.params['b2'] = np.random.randn(1,self.output_size) * np.sqrt(2.0 / self.output_size)
    
    
    # softmax로 각 output class의 확률 값을 예측
    # input > input layer > sigmoid > hiddenLayer > softmax > output
    def predict(self,x):
        if x.ndim == 1: x = x.reshape(1,-1)
   
        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = self.params['W2']
        b2 = self.params['b2']
        
        # 행렬곱이 불가능한 경우.
        if x.shape[1] != w1.shape[0]:
            print("Wrong input shape.. input shape =",x.shape,"Weight =",w1.shape)
            return
      
        z1 = np.dot(x,w1) + b1
        a1 = sigmoid(z1)
        
        z2 = np.dot(a1,w2) + b2
        a2 = softmax(z2)
        
        return a2
          
    # Loss function
    # Cross_entropy_error
    def loss(self,x,t):
        if x.ndim == 1: x = x.reshape(1,-1)
        if(t.ndim == 1) :t = t.reshape(1,-1)         
        if(t.shape[0] != x.shape[0]):
                print("Number of data is different, between x,t !")
                return
            
        res = self.predict(x)
        return cross_entropy_error(res,t)
    
    
    
    # 정확도
    def accuracy(self,x,t):
        if(x.ndim == 1): x = x.reshape(1,-1)
        if(t.ndim == 1) :t = t.reshape(1,-1)
        # 데이터 수가 다를 경우 error
        if(t.shape[0] != x.shape[0]):
                print("Number of data is different, between x,t !")
                return
        
        # 가장 확률이 큰 index를 정답으로 예측
        # target의 해당 index가 정답이 맞는지 count
        predict_idx = self.predict(x).argmax(1)
        trcnt = 0
        for i in range(x.shape[0]):
            if t[i][predict_idx[i]] == 1 : trcnt+=1
       
        # return 맞은 수 / 데이터 수
        return trcnt/x.shape[0]
    
    
    
    # Loss함수의 각 weight의 편미분을 구한다.
    def numerical_gradient(self,x,t,w):
        if w.ndim == 1 : w = w.reshape(1,-1)
        
        h = 1e-4 
        grad = np.zeros_like(w) 
    
        for j in  range(w.shape[0]):
            for i in range(w.shape[1]):
                wi = w[j][i]

                w[j][i] = wi + h
                fw1 = self.loss(x,t)

                w[j][i] = wi - h 
                fw2 = self.loss(x,t) 
        
                # 수치적 미분
                grad[j][i] = (fw1 - fw2) / (2*h)
        
                w[j][i] = wi #원상복귀
        return grad
        
    
    
    # 훈련함수
    # batch 수 만큼 한 step의 학습에 사용한다.
    # 테스트 데이터를 넣게되면, 학습에 사용하지는 않지만, 매 학습마다 테스트 데이터에 대한 평가를 해볼 수 있다.
    def learn(self,X,Y,epochs,batch_size = 10,lr = 0.07,test_X = None,test_Y= None,hist = False,verb=True):
        if(X.ndim == 1): X = X.reshape(1,X.size)
        if(Y.ndim == 1) :Y = Y.reshape(1,Y.size)
        if(X.shape[0] < batch_size) : # batch size가 총 데이터 수보다 클경우 batch_size 수정
            batch_size = X.shape[0]
        
        if hist:
            #hist 초기화
            self.train_accuracylist.clear()
            self.test_accuracylist.clear()
            self.train_losslist.clear()
            self.test_losslist.clear()

        #학습전에 평가를 해본다.
        if verb : print("Before Learning:")
        self.evaluate(X,Y,verb=verb)
        
        # 에폭당 배치 반복 학습 횟수
        batch_per_epoch = X.shape[0] // batch_size
        if X.shape[0] % batch_size != 0 : batch_per_epoch+=1
        
        self.grad = {}
        
        #iter epochs
        for i in range(epochs):
            
            # one epoch Suffle
            # 같은 데이터를 매번 같은 순서로 반복 학습하게 되면 weight가 제대로 학습이 안되므로 전체 데이터 suffle
            suffle_mask = np.random.choice(X.shape[0],X.shape[0],replace=False)
            # batch_size만큼 한 step에 학습.
            x_train = X[suffle_mask]
            y_train = Y[suffle_mask]
            
            # iter step(batch_per_epoch)
            for n in range(batch_per_epoch):  
                start = n*batch_size
                end = min(X.shape[0],(n+1)*batch_size)
                x_train_b = x_train[start:end]
                y_train_b = y_train[start:end]
                
                # batch_size 만큼 데이터로 학습
                self.grad['W1'] = self.numerical_gradient(x_train_b,y_train_b,self.params['W1'])
                self.grad['b1'] = self.numerical_gradient(x_train_b,y_train_b,self.params['b1'])
                self.grad['W2'] = self.numerical_gradient(x_train_b,y_train_b,self.params['W2'])
                self.grad['b2'] = self.numerical_gradient(x_train_b,y_train_b,self.params['b2'])
                
                # Renew params
                self.params['W1'] -= lr*self.grad['W1']
                self.params['b1'] -= lr*self.grad['b1']
                self.params['W2'] -= lr*self.grad['W2']
                self.params['b2'] -= lr*self.grad['b2']
                      
            # 한 에폭마다 평가.    
            if verb: print(i+1)
            self.evaluate(x_train,y_train,hist=hist,verb=verb)
            
            # test data가 있다면 훈련에 사용하지는 않지만, 정확도 추이를 기록한다.
            if (test_X is not None) and (test_Y is not None) :
                    self.evaluate(test_X,test_Y,ev_target="test",hist=hist,verb=verb)
    
    
    
    # data의 정확도와 loss값을 계산하여 출력해준다.
    # 각 학습마다 배열에 기록함으로서 나중에 그래프로 보기 용이하게 한다.
    def evaluate(self,x_data,y_data,ev_target = "train",hist = False,verb = True):
            # Calculate accuracy, loss
            train_accur = self.accuracy(x_data,y_data)
            loss = self.loss(x_data,y_data)
            
            # Select Train hist or Test hist 
            if ev_target == "train":
                target = "| Train_"
                accuracylist = self.train_accuracylist
                losslist = self.train_losslist
            else:
                target = "| Test_"
                accuracylist = self.test_accuracylist
                losslist = self.test_losslist
            
            # Record history
            if hist:
                accuracylist.append(train_accur)
                losslist.append(loss)
            
            # Print
            if verb:
                print(target+"Accuracy:",train_accur,"| LOSS:",loss)
                
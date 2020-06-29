#!/usr/bin/env python
# coding: utf-8

import numpy as np

class KNN:
    def __init__(self,k,x_train,y_train,y_name):
        self.x_train = x_train
        self.y_train = y_train
        self.y_name = y_name
        if k > y_train.shape[0]:
            print("k값이 전체 train data보다 큽니다.\n최대 data 수로 k를 맞춤")
            k = y_train.shape[0]
        
        self.k = k
        # default  weight 계산함수
        self.weight_func = lambda distance: 1 / np.log(distance+2)
        
    def show_dim(self):
        print("Input Demension",self.x_train.shape)
        print("Output Demension",self.y_train.shape)
    
    def get_nearest_k(self,x_test):
        if(x_test.shape[0] != self.x_train.shape[1]):
            print("get_nearest ERROR: test data의 속성 개수가 train된 속성개수와 다릅니다. 하나의 Test Object를 넣어주세요.")
            return;
        
        self.distance = ((x_test - self.x_train)**2).sum(1)   # 모든 점들과의 거리를 계산한 numpy
        self.nearest_idx = (self.distance.argsort())[:self.k] # 가장 가까운 K개의 인덱스를 저장한 nearest_idx
        
        return self.nearest_idx
    
    def obtain_majority_vote(self):
        # 가장가까운 인덱스들 count 후 가장큰 인덱스!!
        vote = np.bincount(self.y_train[self.nearest_idx].reshape(-1))
        
        tvote = np.zeros(len(self.y_name))
        for i in range(len(vote)):
            tvote[i] = vote[i]
            
        self.obtain_vote = tvote;
        return self.y_name[vote.argmax()]
            
    
    def weighted_majority_vote(self):
        vote = np.zeros(len(self.y_name))
        
        # distance를 통해서 weight계산
        weight = self.weight_func(self.distance)
        
        for i in self.nearest_idx:
            vote[self.y_train[i]] += weight[i]
    
        self.weight_vote = vote;
        return self.y_name[vote.argmax()]
    
    
    # numpy of distance를 통해서 weight를 계산하는 함수.
    def set_weight_func(self,func):
         self.weight_func = func
            
    def set_k(self,k):
        self.k = k
 
      



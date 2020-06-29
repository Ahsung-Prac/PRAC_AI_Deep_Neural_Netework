# coding: utf-8
# 2020/인공지능/final/B511209/최아성
import sys, os
import argparse
import time
sys.path.append(os.pardir)

import numpy as np
from AReM import *
from model import *


"""
배치사이즈를 더 크게하고, 모든 data를 표준화로 가공했을 때
Train은 최대 86% Test 78%까지 정확도가 올라갔습니다.
Accuracy 측정방법이 batch_size만큼 총 데이터의 iter를 해서 구했는데,
원 코드 방식대로면, 총데이터의 개수가 batch_size로 나누어 떨어지지 않을때, 나머지 데이터가 측정에 누락되었습니다.
학습시 batch size는 영향이 있지만, 정확도를 측정할때는 상관 없으므로, 한번에 모든데이터를 넣고 측정하는 방식의 코드로 변환했습니다.
test case의 가장 높은 정확도를 보면서 params를 따로 저장해놓으면, 조금더 정확할 수 있으나.
test case에 대해 overfitting 될 수 있으므로, 그렇게 진행은 하지 않습니다.
표준화와 정규화중, 표준화에서 더 높은 정확도를 보여서 model.py에서 predict시 train data에 대한 평균값과 분산값으로 표준화를 하여 데이터를 전처리 했습니다.
batch size = 800개 epochs는 200정도로 학습했을때 결과가 가장 좋았습니다.
"""

class Trainer:
    """
    ex) 200개의 훈련데이터셋, 배치사이즈=5, 에폭=1000 일 경우 :
    40개의 배치(배치당 5개 데이터)를 에폭 갯수 만큼 업데이트 하는것.=
    (200 / 5) * 1000 = 40,000번 업데이트.

    ----------
    network : 네트워크
    x_train : 트레인 데이터
    t_train : 트레인 데이터에 대한 라벨
    x_test : 발리데이션 데이터
    t_test : 발리데이션 데이터에 대한 라벨
    epochs : 에폭 수
    mini_batch_size : 미니배치 사이즈
    learning_rate : 학습률
    verbose : 출력여부

    ----------
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 learning_rate=0.01, verbose=True):
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = int(epochs)
        self.batch_size = int(mini_batch_size)
        self.lr = learning_rate
        self.verbose = verbose
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = int(max(self.train_size / self.batch_size, 1))
        self.max_iter = int(self.epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []


    def train_step(self):
        # 렌덤 트레인 배치 생성
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # 네트워크 업데이트
        self.network.update(x_batch, t_batch)
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            train_acc, _ = self.accuracy(self.x_train, self.t_train)
            test_acc, _ = self.accuracy(self.x_test, self.t_test)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print(
                "=== epoch:", str(round(self.current_epoch, 3)), ", iteration:", str(round(self.current_iter, 3)),
                ", train acc:" + str(round(train_acc, 3)), ", test acc:" + str(round(test_acc, 3)), ", train loss:" + str(round(loss, 3)) + " ===")
        self.current_iter += 1

    def train(self):
        #학습 시작전  정확도 loss
        loss = self.network.loss(self.x_train, self.t_train)
        self.train_loss_list.append(loss)
        train_acc, _ = self.accuracy(self.x_train, self.t_train)
        test_acc, _ = self.accuracy(self.x_test, self.t_test)
        self.train_acc_list.append(train_acc)
        self.test_acc_list.append(test_acc)
        
        for i in range(self.max_iter):
            self.train_step()

        test_acc, inference_time = self.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc) + ", inference_time:" + str(inference_time))

    def accuracy(self, x, t):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0
        start_time = time.time()
        """ 
        주석된 코드의 경우 batch_size로 나누어 남은 나머지만큼
        데이터가 누락되기 때문에 코드를 변경하였습니다.
        """
#         for i in range(int(x.shape[0] / self.batch_size)):
#             tx = x[i * self.batch_size:(i + 1) * self.batch_size]
#             tt = t[i * self.batch_size:(i + 1) * self.batch_size]

#             y = self.network.predict(tx)
#             y = np.argmax(y, axis=1)
#             acc += np.sum(y == tt)
        y = self.network.predict(x)
        y = np.argmax(y, axis=1)
        acc += np.sum(y == t)

        inference_time = (time.time() - start_time) / x.shape[0]

        return acc / x.shape[0], inference_time



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train.py --help 로 설명을 보시면 됩니다."
                                                 "사용예)python train.py --sf=myparam --epochs=10")
    parser.add_argument("--sf", required=False, default="params.pkl", help="save_file_name")
    parser.add_argument("--epochs", required=False, default=200, help="epochs : default=30")
    parser.add_argument("--mini_batch_size", required=False, default=800, help="mini_batch_size : default=100")
    parser.add_argument("--learning_rate", required=False, default=0.01, help="learning_rate : default=0.01")
    args = parser.parse_args()

    #데이터셋 탑재
    (x_train, t_train), (x_test, t_test) = load_AReM(one_hot_label=False)
    # 모델 초기화
    network = Model()
    
    # 트레이너 초기화
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=args.epochs, mini_batch_size=args.mini_batch_size,
                      learning_rate=args.learning_rate, verbose=True)

    # 트레이너를 사용해 모델 학습
    trainer.train()

    # 파라미터 보관
    network.save_params(args.sf)
    print("Network params Saved ")
    
    


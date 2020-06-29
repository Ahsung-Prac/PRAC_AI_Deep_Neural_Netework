# coding: utf-8
# 2020/인공지능/final/B511209/최아성
import sys
import os
import pickle
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append("../../")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))   

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)

        return y.T

    x = x - np.max(x)  # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 시그모이드 활성함수 클래스
class Sigmoid:
    def __init__(self):
        self.out = None
    
    # 순전파
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out
    
    # 역전파
    # 시그모이드의 미분
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


# Relu 활성함수 클래스
class Relu:
    def __init__(self):
        self.mask = None

    # 순전파, 0이하의 값은 모두 0, 나머지는 그대로
    def forward(self, x):
        self.mask = (x <= 0)
        res = x.copy()
        res[self.mask] = 0

        return res
    
    # 역전파, 0으로 바뀐 값들은 미분값이 모두 0
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

#그냥 Relu를 사용하겠습니다.
class CustomActivation:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dout):
        pass

# 행렬곱 계층
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.dW = None
        self.db = None
    
    # 순전파
    def forward(self, x):
        self.original_x_shape = x.shape
        self.x = x
        
        res = np.dot(self.x, self.W) + self.b

        return res

    # 역전파,  W의 미분값은 순전파에서의 입력값(x) * dout
    # 역전파로 전달해야될 미분값은, W * dout
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        return dx


# 출력층 계층, 학습시에만 사용한다.
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None 
        self.y = None 
        self.t = None  
    
    # softmax후 Loss 값을 구해준다.
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
            
        # 정답 레이블이 원-핫 인코딩이 아니라면,
        # numpy모양을 맞춰줘준후, 정답 index는 -1
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    #그냥 lr * 기울기를 뺀다.
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 
            
            
class CustomOptimizer:
    
    """
    Adam
    AdaGrad와 moment의 융합
    가속도와 같은 개념으로 움직이며,
    lr을 점점더 세밀하게 조정해나간다. 
    교재를 참조하여 만들었습니다.
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    # parameters update!!
    def update(self, params, grads):
        if self.m is None:
            
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1 # update 시마다 beta값들 누적곱.. lr_t는 점점 작아진다.
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            # m -> moment 부분
            # v -> AdaGrad 부분
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])         
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            

            
class Model:
    """
    네트워크 모델 입니다.

    """
    def __init__(self, lr=0.01,nomalize = False, standardze = True):
        """
        클래스 초기화
        """
        self.layer_inputSize_list = [6,100,50] # 각 layer의 input size
        self.params = {}
        self.__init_weight()
        self.__init_layer()
        self.optimizer = CustomOptimizer(lr)
        
        self.lr = lr
        self.nomalize_flag = nomalize
        self.standardze_flag = standardze
        
    # 학습 실험할 때  Layer node수를 편리하게 변경하기위한 함수
    def setlist(self,list):
        self.layer_inputSize_list = list.copy()
        self.params = {}
        self.__init_weight()
        self.__init_layer()
        self.optimizer = CustomOptimizer(self.lr) # Optimizer iter를 초기화 하기위해.
    
    def init_layer(self):
        self.__init_layer() # 외부에서 layer를 초기화 하기 위해서.
    
    
    def __init_layer(self):
        """
        affine + activation Func(Relu)
        조합으로 layer 한층,,
        마지막 층은 activation 대신 softmax와 loss 함수
        """
        self.layers = {}
        for idx in range(len(self.layer_inputSize_list)-1):
            self.layers['aff' + str(idx)] = Affine(self.params['W' + str(idx)],self.params['b' + str(idx)])
            self.layers['activ_func' + str(idx)] = Relu()
        
        idx = len(self.layer_inputSize_list)-1
        self.layers['aff' + str(idx)] = Affine(self.params['W' + str(idx)],self.params['b' + str(idx)])

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self):
        n = 2.0
        """
        이 모델은 Relu를 사용하기 때문에 he 초기값 사용
        bias는 0으로 초기화 하는 것이 일반적으로 더 좋다!
        """
        for idx in range(len(self.layer_inputSize_list)-1):
            scale = np.sqrt(n / self.layer_inputSize_list[idx])  
            self.params['W' + str(idx)] = scale * np.random.randn(self.layer_inputSize_list[idx],self.layer_inputSize_list[idx+1])
            self.params['b' + str(idx)] = np.zeros(self.layer_inputSize_list[idx+1])
        
        """마지막 층은 항상 6으로 출력! 마지막은 Relu가 아니기 """
        idx = len(self.layer_inputSize_list)-1
        scale = np.sqrt(1.0 / self.layer_inputSize_list[idx])
        self.params['W' + str(idx)] = scale * np.random.randn(self.layer_inputSize_list[idx],6)
        self.params['b' + str(idx)] = np.zeros(6)
        
    def update(self, x, t):
        """
        gradient를 통해 각 layer들의 기울기를 구하고
        optimizer함수로 weight를 갱신합니다.
        """
        grads = self.gradient(x, t)
        self.optimizer.update(self.params, grads)

    
    def nomalize(self,x):
        """
        test.py에서 처음 정규화가 안되므로,
        Train data에 대한 정규화!
        """
        x= x.astype(np.float32)
        max_el = np.array([56.25, 17.24, 35.  , 11.42, 39.  , 11.09])
        min_el = np.array([0., 0., 0., 0., 0., 0.])
        mean_el = np.array([38.99815725,  1.47474077, 14.33255478,  1.52033287, 15.97623468,
        1.64322051])
        
        return (x - mean_el) / (max_el - min_el)
        
    
    def stabdardz(self,x):
        """
        test.py에서 처음 표준화가 안되므로,
        Train data에 대한 표준화!
        """
        x= x.astype(np.float32)
        mean_el = np.array([38.99815725,  1.47474077, 14.33255478,  1.52033287, 15.97623468,
        1.64322051])
        std_el = np.array([6.31303496, 2.05340679, 5.4094779 , 1.64276685, 6.75185464,
       1.64559124])
        
        return (x-mean_el) / (std_el)
    
    def predict(self, x):
        """
        데이터를 입력받아 정답을 예측하는 함수입니다.
        입력받은 데이터를 전처리후 예측합니다.
        """
        if self.nomalize_flag == True:
            x = self.nomalize(x)
        if self.standardze_flag == True:
            x = self.stabdardz(x)
        
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        데이터와 레이블을 입력받아 로스를 구하는 함수입니다.
        :param x: data
        :param t: data_label
        :return: loss
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)


    def gradient(self, x, t):
        """
        train 데이터와 레이블을 사용해서 그라디언트를 구하는 함수입니다.
        첫번째로 받은데이터를 forward propagation 시키고,
        두번째로 back propagation 시켜 grads에 미분값을 리턴합니다.
        :param x: data
        :param t: data_label
        :return: grads
        """
        # forward, 입력값들 저장한다.
        self.loss(x, t)

        # backward, 오차역전파로 기울기를 구한다.
        dout = 1
        dout = self.last_layer.backward(dout)

        # 역전파는 layer를 역순으로 전파한다.
        rev_layers = list(self.layers.values())
        rev_layers.reverse()
        
        for layer in rev_layers:
            dout = layer.backward(dout)

        #  기울기 저장!
        grads = {}
        for idx in range(len(self.layer_inputSize_list)):
            grads['W' + str(idx)] = self.layers['aff' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['aff' + str(idx)].db

        return grads
    

    def save_params(self, file_name="params.pkl"):
        """
        네트워크 파라미터를 피클 파일로 저장하는 함수입니다.

        :param file_name: 파라미터를 저장할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        """
        저장된 파라미터를 읽어와 네트워크에 탑재하는 함수입니다.

        :param file_name: 파라미터를 로드할 파일 이름입니다. 기본값은 "params.pkl" 입니다.
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        
        self.__init_layer() # load 한 params를 layer에 다시 초기화 합니다.!!

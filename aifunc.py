import numpy as np

# 계단함수 구현
def step_function(x,bound = 0):
    a = np.array(x>bound) # 0보다 큰값은  true 나머지는 false로 배정
    return a.astype(np.int) #int 타입으로 변경 0 or 1

#시그모이드
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

def relu(x):
    return np.maximum(0,x)

# 데이터 여러개일 경우 softmax
def softmax(x):
    if x.ndim == 1 : x = x.reshape(1,-1)
    exp_a = np.exp(x)
    # np 스칼라 확장을 위해, shape를 맞춰준다.
    # 한 행마다 각 행의 총합으로 나누어 져야한다.
    sum_exp_a = np.sum(exp_a,axis = 1).reshape(-1,1)
    return exp_a/sum_exp_a

# loss 함수
def cross_entropy_error(y, t):
    epsilon = 1e-7

    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)

    batch_size = y.shape[0]

    cee = np.sum(-t * np.log(y + epsilon)) / batch_size
    return cee

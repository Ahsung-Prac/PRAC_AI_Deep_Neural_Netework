import matplotlib.pyplot as plt
import numpy as np

# 계단함수 구현
def step_function(x):
    a = np.array(x>0) # 0보다 큰값은  true 나머지는 false로 배정
    return a.astype(np.int) #int 타입으로 변경 0 or 1


x = np.arange(-5.0,5.0,0.1)
y = step_function(x)

plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.xlim(-6,6)
plt.ylim(-0.1,1.1)
plt.show()

def relu(x):
    return np.maximum(0,x)

x = np.arange(-5.0,5.0,0.1)
y = relu(x)

plt.plot(x,y)
plt.show()

x2 = np.array([-1,4,7,-9])
def softmax(x):
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
   
    return exp_a/sum_exp_a

print(softmax(x2))
print(np.sum(softmax(x2)))
    

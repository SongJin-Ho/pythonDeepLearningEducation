import numpy as np
import matplotlib.pyplot as plt

'''
계단 함수
numpy 배열을 인수로 넣을 수는 없습니다.
ex) step_finction(np.array([1.0,2.0]))
'''
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function_np(x):
    y = x > 0
    return y.astype(np.int) #[1,1]
    # return y (True True)


A = np.array([1.0,2.0])
# print(step_function(A)) # error 발생 : ValueError: The truth value of an array with more than one element is ambiguous
print(step_function_np(A))

#y는 현재 bool 형식 list로 구성되지만 astype(np.int) 를 이용하여 결과를 정수로 출력해준다.
x = np.array([-1.0,1.0,2.0])
y = x > 0
print(y)
print(y.astype(np.int))

x = np.arange(-5.0, 5.0, 0.1) # -5.0 ~ 5.0까지 0.1 간격
y = step_function_np(x) # y = h(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()


#시그모이드 함수 구현
def sigmoid(x):
    return 1 / (1+np.exp(-x))

y = sigmoid(x)
plt.clf()
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

#ReLU 함수
def relu(x):
    return np.maximum(0,x)

y = relu(x)
plt.clf()
plt.plot(x,y)
plt.show()

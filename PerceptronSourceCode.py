import numpy as np


def print_function(input_value):
    print(input_value)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


'''
3층 신경망 구현하기
'''

X = np.array([1.0, 0.5])  # (2,)
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # (2,3)
B1 = np.array([0.1, 0.2, 0.3])  # (3,)

print_function(W1.shape)
print_function(X.shape)
print_function(B1.shape)

A1 = np.dot(X, W1) + B1  # ((2,) X (2,3))+ (3,)
Z1 = sigmoid(A1)  # (3,)

print_function(A1)  # (3,)
print_function(Z1)  # (3,)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])  # (3,2)
B2 = np.array([0.1, 0.2])  # (2,)

print_function(W2.shape)
print_function(B2.shape)

A2 = np.dot(Z1, W2) + B2  # ((3,) X (3,2)) + (2,)
Z2 = sigmoid(A2)  # (2,)

print_function(A2)
print_function(Z2)

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])  # (2,2)
B3 = np.array([0.1, 0.2])  # (2,)

A3 = np.dot(Z2, W3) + B3  # (2,) X (2,2) + (2,)
Z3 = identity_function(A3)

print_function(A3)
print_function(Z3)


'''
위의 code들을 간단하게 작성하면 다음과 같다
'''
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([[0.1,0.2,0.3]])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([[0.1,0.2]])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network,x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print_function(y)

'''
출력층의 활성화 함수는 어떤 문제냐에 따라 달라진다.
1) 회귀에서는 항등 함수 ex) 위에서 나와있는 identity_function(x)
2) 분류에서는 소프트맥스 함수를 사용한다. ex) y_k= exp(x_k) / (n ∑i=1) exp(a_i)

exp(x)는 지수 함수를 뜻하며, n은 출력층의 뉴런 수, y_k는 그중 k번째 출력임을 뜻한다.
'''

#softmax 함수
a = np.array([0.3,2.9,4.0])
exp_a = np.exp(a) # 분자 exp_a
sum_exp_a = np.sum(exp_a)
print_function(exp_a)
print_function(sum_exp_a)

y = exp_a / sum_exp_a
print_function(y)

def softmax(x):
    c = np.max(x) #max 부분은 밑에 소프트맥스 함수 구현시 주의점을 읽은뒤에 진행하도록 한다.
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

print_function(softmax(a))

'''
소프트맥스(softmax) 함수 구현시 주의점
1) softmax 함수는 지수 함수를 사용하기 때문에 아주 큰 값이 나올수도 있다.
2) 이를 방지하기 위해서 일반적으로 입력 신호중 최댓값을 이용하여 오버플로를 방지한다.
'''

#1) example

print_function(np.exp(10))
print_function(np.exp(100))
# print_function(np.exp(1000)) #오버플로우 발생

#2) example
a = np.array([1010,1000,990])
# print_function(np.exp(a) / np.sum(np.exp(a)))
c = np.max(a)
print_function(np.exp(a-c) / np.sum(np.exp(a-c)))

'''
소프트맥스(softmax)의 특징은 0~1 사이의 실수값으로 나타난다.
그러므로 총 합은 1이 된다.
'''
a = np.array([0.3,2.9,4.0])
y = softmax(a)
print_function(y)
print_function(np.sum(y))

'''
기계학습의 문제 풀이는 학습과 추론의 두 단계를 거쳐 이뤄지는데,
1) 추론단계에서는 출력층의 소프트맥스 함수를 생략하는 것이 일반적이며,
2) 학습단계에서는 출력층에 소프트맥스 함수를 사용한다.
'''
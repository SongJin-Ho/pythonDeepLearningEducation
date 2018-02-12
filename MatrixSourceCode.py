import numpy as np


def print_function(input_value):
    print(input_value)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def identity_function(x):
    return x

A = np.array([1,2,3,4])
B = np.array([[1,2],[3,4]])
C = np.array([[5,6],[7,8]])

#np.ndim()은 배열의 차원 수
#shape()은 배열의 형상 ※ shape은 튜플로 반환됨

print_function(np.ndim(A))
print_function(A.shape)
print_function(B.shape)

'''
두 행렬의 내적(행렬의 곱)
행렬의 내적은 1번째 차원의 원소 수(열 수)와 행렬 B의 0번째 차원의 원소 수(행 수)가 같아야 한다.
ex) A = 2(행)x3(열) B = 3(행) x 2(열) 은 가능하나,
    A = 3(행)x2(열) B = 3(행) x 2(열) 은 불가능하다.
'''
print_function(np.dot(B,C))

'''
A가 2차원 행렬, B가 1차원 배열일 때도 대응하는 차원의 원소 수를 일치해야함
'''
A = np.array([[1,2],[3,4],[5,6]])
print_function(A.shape)
B = np.array(([7,8]))
print_function(B.shape)
print_function(np.dot(A,B))

'''
신경망도 똑같은 방법으로 구현할 수 있음
'''
X = np.array([1,2])
print_function(X.shape)
W = np.array([[1,3,5,7],[2,4,6,8]])
print_function(W.shape)
print_function(np.dot(X,W))



import numpy as np

# 1차 행렬 = vector
vector = np.array([1.0,2.0,3.0])
print(vector)

# 2차 행렬 = matrix
matrix = np.array([[1,2],[3,4]])
print(matrix)

#브로드 캐스트(broadcast)
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([10,20])

print(A * B)

#원소 접근
for row in A:
    for index in row:
        print(index)

#1차원 배열로 변환(평탄화)
print(A.flatten())

#인덱스 0,2,4인 원소 얻기
X = A.flatten()
print(X[np.array([0,2,4])])

# boolean, 조건
BX = X>15
print(BX,BX.dtype)
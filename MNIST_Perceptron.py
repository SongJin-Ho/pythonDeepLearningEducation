import sys, os
import numpy as np

'''
MNIST Data를 이용하여 신경망을 구현해보도록 한다.
MNIST의 이미지 데이터는 28x28 크기의 회색조 이미지이며, 각 픽셀은 0에서 255까지의 값을 취한다.
'''
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

'''
normalize : 0.0 ~ 1.0 사이로 정규화 할지 여부, False = 0 ~ 255
flatten : 1차원 배열로 만들지 여부. False 일 경우 : 1x28x28의 3차원 배열로 나타내줌
one_hot_label : 정답을 뜻하는 원소만 1, 나머지는 0으로 설정할지 여부
'''
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize= False)

print(x_train.shape) # (60000, 784) 6만개의 데이터, 28x28 : 784 크기
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)

from PIL import Image
import pickle

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

# img_show(img)

'''
이제 신경망을 이용해서 결과를 추론해보겠다.
이미지의 크기는 28x28 = 784이며 출력층은 0~9까지 이미지이므로 총 10개이다.
은닉층은 임의로 설정하여 1층은 50개, 2층은 100개로 진행한다.
'''

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    print(x_test.shape)
    print(t_test.shape)
    return x_test,t_test

def init_network():
    with open("./dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(x):
    c = np.max(x) #max 부분은 밑에 소프트맥스 함수 구현시 주의점을 읽은뒤에 진행하도록 한다.
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def predict(network,x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y



x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

'''
nomalize를 True로 변경하여 0~255 범위인 각 픽셀의 값을 0.0 ~ 1.0 범위로 변환하였다. (단순히 픽셀의 값을 255로 나눈거임)
이처럼 데이터를 특정 범위로 변환하는 처리를 '정규화'라 하고 특정 변환을 가하는 것을 '전처리'라 한다.
전처리는 학습 속도를 높이는 등을 많이 진행하며 데이터 전체평균과 표준편차를 이용하여 데이터들이 0을 중심으로 분포하도록 이동시키거나
데이터의 확산 범위를 제한하는 정규화를 수행한다.
전체 데이터를 균일하게 분포시키는 데이터 백색화 등도 있다.
'''

'''
배치 처리 방식
'''
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

'''
한꺼번에 처리
'''

x, t = get_data()
network = init_network()

accuracy_cnt = 0
y = predict(network,x)
p = np.argmax(y,axis=1)
accuracy_cnt += np.sum(p == t)

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
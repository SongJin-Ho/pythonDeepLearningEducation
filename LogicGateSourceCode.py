import numpy as np


def AND(input1, input2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = input1 * w1 + input2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1


'''
    AND(0,0) => 0*0.5 + 0*0.5 <= 7
    AND(0,1) => 0*0.5 + 1*0.5 <= 7
    AND(1,0) => 1*0.5 + 0*0.5 <= 7
    AND(1,1) => 1*0.5 + 1*0.5 <= 7
'''


def OR(input1, input2):
    w1, w2, theta = 0.5, 0.5, 0.5
    tmp = input1 * w1 + input2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1


'''
    OR(0,0) => 0*0.5 + 0*0.5 <= 5
    OR(0,1) => 0*0.5 + 1*0.5 <= 5
    OR(1,0) => 1*0.5 + 0*0.5 <= 5
    OR(1,1) => 1*0.5 + 1*0.5 <= 5
'''


def NAND(input1, input2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = input1 * w1 + input2 * w2
    if tmp <= theta:
        return 1
    else:
        return 0


'''
    NAND(0,0) => 0*0.5 + 0*0.5 <= 7
    NAND(0,1) => 0*0.5 + 1*0.5 <= 7
    NAND(1,0) => 1*0.5 + 0*0.5 <= 7
    NAND(1,1) => 1*0.5 + 1*0.5 <= 7
'''


# 가중치와 편향 도입
# w : 가중치 b : 편향
def AND_BIAS(input1, input2):
    x = np.array([input1, input2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR_BIAS(input1, input2):
    x = np.array([input1,input2])
    w = np.array([0.5, 0.5])
    b = -0.4
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1

'''
    (0,0) = 0*0.5 + 0*0.5 - 0.4 <= 0 TRUE
    (0,1) = 0*-0.5 + 1*-0.5 - 0.4 <= 0 FALSE
    (1,0) = 1*-0.5 + 0*-0.5 - 0.4 <= 0 FALSE
    (1,0) = 1*-0.5 + 1*-0.5 - 0.4 <= 0 FALSE    
'''

def NAND_BIAS(input1, input2):
    x = np.array([input1, input2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(x*w) + b
    if tmp <= 0:
        return 0
    else:
        return 1
'''
    (0,0) = 0*-0.5 + 0*-0.5 + 0.7 <= 0 FALSE
    (0,1) = 0*-0.5 + 1*-0.5 + 0.7 <= 0 FALSE
    (1,0) = 1*-0.5 + 0*-0.5 + 0.7 <= 0 FALSE
    (1,0) = 1*-0.5 + 1*-0.5 + 0.7 <= 0 TRUE    
'''

def XOR(input1, input2):
    return AND_BIAS(NAND_BIAS(input1,input2),OR_BIAS(input1,input2))

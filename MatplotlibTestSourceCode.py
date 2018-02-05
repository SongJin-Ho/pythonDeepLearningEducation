import matplotlib.pyplot as plt
import numpy as np

#sin 함수
x = np.arange(0,6,0.1) # 0에서 6까지 0.1 간격으로 생성
y = np.sin(x) #sin 함수 생성

plt.plot(x,y)
plt.show()

#cos 함수

y1 = np.sin(x)
y2 = np.cos(x)

# linestyle, label 사용
plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle="--",label="cos") #cos 함수는 점선으로 그리기
plt.xlabel("x") #x축 이름
plt.ylabel("y") #y축 이름
plt.title("sin & cos") #제목
plt.legend()
plt.show()

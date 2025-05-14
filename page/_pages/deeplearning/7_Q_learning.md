---
layout: archive
title: " chapter.7 신경망과 Q러닝 "
permalink: /coding/deeplearning/Q/ 
author_profile: true
math: true
---

Index

1. DeZero
    - Dezero
    - 역전파
    - 다차원배열 Tensor
    - 최적화
2. 선형회귀
3. 신경망
    - 활성화함수
    - 계층과 모델
4. Q러닝과 신경망

지금까지의 환경모델은 3 X 4 그리드월드에서 상태가 12, 행동이 4개였기 때문에 Q함수의 가짓수는 48개이다. 하지만 현실에서 상태와 행동이 무수히 많은 경우, 이 데이터를 딕셔너리로 (또는 테이블) 관리하도록 구현하는 것은 한계가 존재한다.  
체스로 예를들면 체스보드의 패턴이 10^123 으로 상태가 이 수만 존재한다. 이렇게 막대한 수를 모두 경험하는 것은 현실적으로 불가능  

따라서 Q함수를 더 가벼운 함수로 근사하는 방법 중 유력한 딥러닝을 지금까지의 강화학습에 결합하여 심층 강화학습. _(딥러닝을 도구로 이용)_

---

### 1. DeZero

⑴ Dezero

 pytorch를 기반으로 설계된 딥러닝 프레임워크

```python
import numpy as np
from dezero import Variable

x_np = np.array(5.0)

x = Variable(x_np)

y = 3 * x ** 2 # variable instace

print(y)  # variable(75.0)
```

Numpy 배열은 1차원 ~ n차원 등 여러 형태로 데이터 표현 가능하여 python에서 수치계산을 빠르게 할 수 있도록 도와주는 자료구조로 행렬이나 벡터 등의 수학적 개념을 다룰 수 있다.  

Variable은 Dezero에서 가져온 클래스로 numpy 배열을 _감싸는_ 역할을 한다. numpy를 그대로 쓰지않고 Variable 인스턴스로 관리하는 이유는 **자동미분**이 가능하도록 하기 위해서이다.  
일반 numpy 배열은 어떤 연산을 거쳤는지에 대한 기록이 남지 않으며 딥러닝에서 중요한 가중치 미분 과정이 자동으로 되지 않기 때문에 Variable 클래스를 이용하는 것이다.  
이때, 모든 연산이 기록되어 **역전파**와 **자동미분**이 가능해진다.


<div style="border: 1px solid black; padding: 10px; margin-bottom: 10px;">
    <strong>일반 numpy 배열</strong><br>
    ──────────────────────────<br>
    np.array(5.0)<br>
    
</div>

<div style="border: 1px solid black; padding: 10px; margin-bottom: 10px;">
    <strong>Variable 클래스</strong><br>
    ──────────────────────────<br>
    데이터: 5.0<br>
    미분 정보: 추적<br>
    연산 기록: 0<br>
</div>

⑵ 역전파 Backpropagation  

딥러닝 모델은 예측값과 실제값의 차이를 줄여나가야 하고 이때 미분으로 오차를 계산하여 가중치를 수정하는 과정을 거치게 된다.

순전파 Fowardpropagation 은 입력 데이터를 받아서 예측값을 계산 

[신경망 흐름]
1. 순전파로 예측값 생성  
2. 손실함수 ( Loss Function =  (y - t) ** 2 / t = 실제값) 으로 예측값과 실제값 비교  
3. 역전파를 이용해 손실 기준 미분 계산 ( ∂L / ∂y == 2( y - t) ..)
4. 가중치 업데이트

```python
y.backward()
print(x.grad) 
```

⑶ 다차원 배열 Tensor 

 1D : vector
 2D : matrix
 3D : tensor

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/tensor.png" alt="tensor" width="500">
</div>  

다차원배열은 원소의 배열에 방향이 있으며 그 방향을 차원 or 축 이라고 한다.

- 벡터의 내적 Dot product

벡터의 내적은 두 벡터를 곱해서 하나의 스칼라값을 얻는 연산

a * b = a_1 * b_1 + a_2 * b_2 + ... + a_n * b_n

따라서 두 벡터가 같은 방향(부호가 같을때) 값이 커지고 반대일때는 값이 작아지는 특징이 있으며, 두 벡터가 직교하는 경우 내적은 0이 된다.  

- 행렬의 곱

행렬의 곱에서 벡터의 내적이 사용되며 _왼쪽 행렬 가로벡터_ 와 _오른쪽 행렬 세로벡터_ 사이의 내적을 계산하면 결과값이 새로운 행렬의 해당 원소가 된다.

**⇒ 내적하는 행렬끼리는 차원의 원소수가 일치해야 계산이 이루어질 수 있다.**

python에서 _numpy.dot()_ 함수를 사용해 내적 계산하며 신경망 학습에서 주로 가중치 w와 입력값의 곱으로 사용하게 되며 역전파 단계에서 미분을 통한 경사 계산에서 내적을 사용한다. 

```python

```

⑷ 최적화

로젠브록 함수 rosenbrock function  
이 함수의 출력값이 최소가 되는 x_0과 x_1을 찾는 것이 최적화릐 목표로, Dezero로 이를 찾는 방법을 알아볼 것.

$$
y = 100(x_1 - x_0^2)^2 + (x_0 - 1)^2
$$


<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/rosenbrock_function.png" alt="rosenbrock_function" width="500">
</div>  

```python
import numpy as np
from dezero import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

iters = 10000  # 반복 횟수
lr = 0.001     # 학습률

for i in range(iters):  # 갱신 반복
    y = rosenbrock(x0, x1)

    # 이전 반복에서 더해진 미분 초기화
    x0.cleargrad()
    x1.cleargrad()

    # 미분(역전파)
    y.backward()

    # 변수 갱신
    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

print(x0, x1)
```

- 경사하강법 Gradient Descent  

경사하강법은 최적화 알고리즘 중 하나. 함수의 최소 or  최대값을 찾기 위한 방법. 

경사는 어떤 함수의 기울기를 의미하며 수학적으로 함수 f(x)의 경사는 미분으로 정의된다. 따라서 위 코드에서 x0과 x1의 미분값을 모아 만든 벡터는 기울기 벡터가 된다. 
이 기울기 벡터는 단순히 미분값을 모아놓은 벡터로, 벡터가 가장 가파르게 오르는 방향을 가리킨다. 벡터의 내적에서 설명했듯이 _벡터의 내적은 두 벡터가 얼마나 같은 방향으로 가고 있는지를 나타내기 때문에_ 각 지점에서 함수의 출력을 가장 크게 증가시키는 방향을 가리킨다.  
(산을 생각하면 가장 가파른 코스가 가장 빠르게 도착한다.)  
**따라서 각 지점에서 y를 가장 크게 감소시키기 위해서는 기울기에 - 를 곱하면 된다.** 이것이 경사하강법이다. 

```python
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F

# 토이 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)  # 생략 가능

# 매개변수 정의
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

# 예측 함수
def predict(x):
    y = F.matmul(x, W) + b  # 행렬의 곱으로 여러 데이터 일괄 계산
    return y

```
lr : 학습률

---

### 2. 선형 회귀 Linear Regression

머신러닝은 문제 해결 방법 자체를 컴퓨터가 수집된 데이터에서 해결책을 찾는 것이다. 선형회귀는 머신러닝의 가장 기본이다.  


```python

```

토이데이터셋은 x와 y 선형관계가 있지만 노이즈가 껴있다. 노이즈는 데이터를 측정하거나 수집할때 생기는 불확실성 및 오차를 의미한다. 실제로는 정확한 직선이어야할 데이터가 조금 흩어진 형태로 나타나는게 노이즈가 껴있다는 것을 의미한다. 실제로 데이터를 수집할때도 이러한 오차가 발생하기 때문에 모델이 학습을 할때 흩어진 데이터들을 일반화할 수 있도록 학습시키는 과정을 거침  

- 선형회귀이론

선형회귀는 y와 x가 선형관계라고 가정하기 때문에 y = W * x + b 로 표현된다. 
W : 기울기 (scala)  
x : 입력데이터  
b : 절편 bias  

- 잔차 Residual

y는 실제 데이터이며 예측값 W*x + b 사이의 차이  
이 잔차를 최소화하기 위한 학습을 진행하게 되는데 이때 **잔차 제곱의 평균을 손실함수**라고 하며 이를 최소화  

$$
L = \frac{1}{N} \sum_{i=1}^N (W x_i + b - y_i)^2
$$


```python
import numpy as np
from dezero import Variable
import dezero.functions as F

# 토이 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)  # 노이즈가 추가된 직선

# 매개변수의 정의
W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

# 예측 함수
def predict(x):
    y = F.matmul(x, W) + b
    return y

```

```python
# 평균 제곱 오차(식 7.2) 계산 함수
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

# 경사 하강법으로 매개변수 갱신
lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    # 또는 loss = F.mean_squared_error(y, y_pred)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0:  # 10회 반복마다 출력
        print(loss.data)

print('====')
print('W =', W.data)
print('b =', b.data)

# [그림 7-9] 학습 후 모델
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(t)
plt.plot(t, y_pred.data, color='r')
plt.show()
```

### 3. 신경망 

복잡한 데이터셋이라면? 

선형이 아닌 비선형의 데이터셋은 선형회귀로 대응할 수 없기때문에 _신경망_ 을 사용  

```python
import numpy as np

np.random.seed(0)
x = np.random.rand(100,1)
y = np.sin(2 * np.pi * x) + np.random.rand(100,1)
```

sin() 함수를 통해 비선형의 데이터셋 준비

- 선형회귀 핵심코드  _행렬 곱, 덧셈_
    y = F.matmul(x, w) + b 

- 선형변환 or 어파인변환(엄밀히 말하면 곱까지만)
    y = F.linear(x, W, b)

    선형변환은 신경망의 모든 노드가 다음층의 모든 노드와 연결되어 상호작용되는 _완전연결계층(Fully Connected Layer)_


⑴ 활성화 함수

$$
y = σ(W*x + b)
$$

선형 변환의 경우 입력 데이터에 대한 선형적 변환을 수행하지만 신경망은 선형변환의 출력을 비선형으로 변환한다.  
**활성화함수는 비선형 변환을 수행한다.**  

- 왜 비선형 변환을 사용하는가?

    직선형태의 데이터셋의 경우, 직선으로 데이터를 구분할 수 있으나 복잡한 데이터셋은 직선으로는 구분이 어렵다. 따라서 비선형 변환을 거치는 것. 

- 시그모이드 함수 sigmoid  

    데이터를 S자 곡선 형태로 압축  **F.sigmoid()**  
$$
y = \frac{1}{1 + e^{-x}}
$$

- ReLU (Recified Linear Unit)

    : 양수는 그대로 두고, 음수는 0으로 만들어서 통과할 수 있도록 특정 뉴런은 비활성화 **F.relu()**


$$
y = 
\begin{cases} 
0 & \text{if } x < 0 \\ 
x & \text{if } x \ge 0
\end{cases}
$$

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/ReLU.png" alt="tanh" width="500">
</div>

- Tanh

    : -1과 1 사이로 값을 압축하여 양극화 표현이 가능

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/tanh.png" alt="tanh" width="500">
</div>

- 구현  

일반적인 신경망은 선형변환과 활성화함수를 번갈아 사용  

```python
import numpy as np
from dezero import Variable
import dezero.functions as F

# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 매개변수 초기화
I, H, O = 1, 10, 1  # I=입력층 차원 수, H=은닉층 차원 수, O=출력층 차원 수
W1 = Variable(0.01 * np.random.randn(I, H))  # 첫 번째 층의 가중치
b1 = Variable(np.zeros(H))                   # 첫 번째 층의 편향
W2 = Variable(0.01 * np.random.randn(H, O))  # 두 번째 층의 가중치
b2 = Variable(np.zeros(O))                   # 두 번째 층의 편향

# 신경망 추론
def predict(x):
    y = F.linear(x, W1, b1)      # 선형 변환
    y = F.sigmoid(y)             # 활성화 함수 (시그모이드 함수 사용)
    y = F.linear(y, W2, b2)      # 선형 변환
    return y

lr = 0.2
iters = 10000

# 신경망 학습 (매개변수 갱신)
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)
    
    W1.cleargrad() #기울기 초기화
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    
    loss.backward()
    
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    
    if i % 1000 == 0:  # 1000회 반복마다 출력
        print(loss.data)
```

 I=입력층 차원 수, H=은닉층 차원 수, O=출력층 차원 수  

⑵ 계층과 모델

위의 코드를 더 쉽게 작성할 수 있도록 도와주는 Dezero의 모듈이 있다.  

- Linear

    : Linear 클래스는 신경망에서 선형변환을 수행하는 계층

    Linear(out_size, nobias=False, dtype=np.float32, in_size=None)

    out_size : 출력크기 (출력 데이터의 차원 수)
    nobias : 편향사용여부 (b)
    dtype : 입력데이터유형
    in_size : 입력데이터의 차원수

```python
import numpy as np
from dezero import Model
import dezero.layers as L
import dezero.functions as F

# 데이터셋 생성
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2   # 학습률
iters = 10000  # 반복 횟수

# 신경망 모델 정의
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

# 모델 생성
model = TwoLayerNet(10, 1)

# 학습
for i in range(iters):
    y_pred = model.forward(x)  # 또는 y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()   # 기울기 초기화
    loss.backward()      # 역전파
    for p in model.params():   # 모델의 모든 파라미터에 대해 업데이트
        p.data -= lr * p.grad.data

    if i % 1000 == 0:  # 1000번마다 로스 출력
        print(loss)
```

- 최적화기법 

```python
import numpy as np
from dezero import Model
from dezero import optimizers  # 옵티마이저 불러오기
import dezero.layers as L
import dezero.functions as F

# 데이터셋 생성
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# 신경망 모델 정의
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y

# 모델 생성 및 옵티마이저 설정
model = TwoLayerNet(10, 1)
optimizer = optimizers.SGD(lr=0.2)  # 학습률 0.2 설정
optimizer.setup(model)  # 옵티마이저에 모델 등록

# 학습
iters = 10000
for i in range(iters):
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y, y_pred)
    
    model.cleargrads()
    loss.backward()
    optimizer.update()  # 옵티마이저가 자동으로 업데이트

    if i % 1000 == 0:
        print(loss)
```

불러온 옵티마이저 중 하나인 SGD 생성  
SGD는 _확률적 경사하강법_ 이라는 최적화 기법 구현. 매개변수를 기울기 방향으로 lr배만큼 갱신. 
(확률적 경사하강법이란 확률적 즉, 무작위 적으로 데이터를 선택하여 경사 하강법을 수행하는 기법으로 딥러닝에서 매우 흔히 쓰이는 최적화 기법)

---

### 4. Q러닝과 신경망

다시한번 복습하자면 TD에서 배운 SARSA는 현재 상태에서 행동을 선택하고 그에 대한 보상을 받은 후 그 다음의 상태와 그때의 행동까지를 구해서 현재 상태의 Q함수를 업데이트이며 On-poicy이다.  
Q러닝은 SARSA의 off-policy 버전으로 생각할 수 있다. 대신 보정을 따로 하지 않고, 실제 선택한 행동을 기준으로 업데이트하는 SARSA와 달리 가장 좋은 Q값을 만드는 행동 즉 최적의 행동으로만 업데이트를 한다는 차이점이 있다. 

⑴ 신경망 전처리  

- 원핫벡터 one-hot vector  

신경망에서 범주형 데이터를 처리할 때는 원핫벡터로 변환하는 것이 일반적.  

ex )  여러 원소 중 하나만 1이고 나머지는 모두 0인 벡터
S -> (1, 0, 0)  
M -> (0, 1, 0)  
L -> (0, 0, 1)  

현재 3 X 4 의 그리드월드는 (x,y) 좌표로 표현한다. 이는 연산가능한 수치가 아닌 범주형 데이터이며 아래는 이를 전처리하는 코드이다.  

- Q함수 표현 신경망  
```python
import numpy as np

def one_hot(state):
    # 3×4 그리드 월드의 12차원 벡터 생성 - 0으로 초기화할
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)

    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0

    return vec[np.newaxis, :] #벡터에 차원을 추가

state = (2, 0)
x = one_shot(state) #state를 받아서 원핫벡터로 변환

print(x.shape)
print(x)
```

```python
from dezero import Model
import dezero.functions as F
import dezero.layers as L

class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)     # 중간층의 크기
        self.l2 = L.Linear(4)       # 행동의 크기 (가능한 행동의 개수)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

qnet = QNet()

state = (2, 0)

# 원-핫 벡터로 변환
state = one_hot(state)  
qs = qnet(state)
print(qs.shape)   # [출력 결과] (1, 4)

```

⑵ 신경망과 Q러닝

- Q러닝

$$
Q'(S_t, A_t) = Q(S_t, A_t) + \alpha \left[ R_t + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$

목표(스칼라값)
$$
T  ⇒ R_t + \gamma \max_a Q(S_{t+1}, a)
$$

$$
Q'(S_t, A_t) = Q(S_t, A_t) + \alpha \left[T - Q(S_t, A_t) \right]
$$


```python
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
import matplotlib.pyplot as plt
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from common.gridworld import GridWorld


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)  # 중간층의 크기
        self.l2 = L.Linear(4)    # 행동의 크기(가능한 행동의 개수)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()                        # 신경망 초기화
        self.optimizer = optimizers.SGD(self.lr)  # 옵티마이저 생성
        self.optimizer.setup(self.qnet)           # 옵티마이저에 신경망 등록

    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        # 다음 상태에서 최대가 되는 Q 함수의 값(next_q) 계산
        if done:  # 목표 상태에 도달
            next_q = np.zeros(1)  # [0.]  # [0.] (목표 상태에서의 Q 함수는 항상 0)
        else:     # 그 외 상태
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)
            next_q.unchain()  # next_q를 역전파 대상에서 제외

        # 목표
        target = self.gamma * next_q + reward
        # 현재 상태에서의 Q 함수 값(q) 계산
        qs = self.qnet(state)
        q = qs[:, action]
        # 목표(target)와 q의 오차 계산
        loss = F.mean_squared_error(target, q)

        # 역전파 → 매개변수 갱신
        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data


env = GridWorld()
agent = QLearningAgent()

episodes = 1000  # 에피소드 수
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state

    average_loss = total_loss / cnt
    loss_history.append(average_loss)


# [그림 7-14] 에피소드별 손실 추이
plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# [그림 7-15] 신경망을 이용한 Q 러닝으로 얻은 Q 함수와 정책
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)
```

- 에피소드별 평균 손실  
<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/Q3_loss.png.png" alt="Q3" width="500">
</div>

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/Q4.png" alt="Q4" width="500">
  <img src="/assets/images/Q5.png" alt="Q5" width="500">
</div>
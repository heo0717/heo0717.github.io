---
layout: archive
title: " Chapter.9 정책경사법"
permalink: /coding/deeplearning/9/ 
author_profile: true
math: true
---

## 정책경사법

Index

1. 정책경사법
    - 가치기반기법
    - 정책기반기법 도출
    - 정책경사법 알고리즘
2. Reinforce
3. Baseline
    - 아이디어
    - 적용
4. 행위자 - 비평자
5. 정리

---  

밴디트, 벨만, DP, MC, TD   
Q-learning, DQN  
            정책경사법

---  

강화학습에서 정책을 얻는 방법은 Q함수값으로 정책을 결정하는 Q러닝 계열과 정책 그 자체를 평가하는 방법 두가지로 나뉜다. 

### 정책경사법 Policy Gradient Method

##### ⑴ 가치 기반 기법 
: 가치함수(V / Q)를 학습하고 이를 정책의 평가와 개선을 반복하여 최적의 행동을 찾아 최적 정책을 찾아간다. 즉, **가치함수 경유** 하여 정책을 얻는다.  

- 가치 기반 기법의 한계 
1. Q값을 통해 최적의 행동을 선택하기 때문에 정책을 직접 조절하지 못한다
2. 행동이 연속적이면 Q함수로 표현이 어렵다 
3. epsilon-greedy 는 확률적 정책 학습이 어렵다

    π(s) = argmax_a Q(s, a)

    argmax로 제일 좋은 행동 하나를 선택하는 결정적 정책을 따르는데, 여러 행동 중에서 확률적으로 선택하는 방식이 _확률적 정책_

    4. Q함수의 추정이 잘못된 경우 정책 또한 불안정해진다

##### ⑵ 정책경사법
경사를 이용하여 정책을 갱신하는 기법의 총칭  
정책을 직접 학습, 조정해서 기대보상을 최대화하는 기법

- 특징

1. 정책을 직접 학습한다
2. **연속적인 행동 학습 가능** (ex. 가속을 '얼마나' 밟을 것인지)
3. 확률정책 가능
4. 미분 가능한 정책을 경사로 학습 가능  

이산적 행동 - 가치기반기법  
연속적 행동 - 정책경사법  
 
---

정책경사법의 목적 : 좋은 정책을 학습해서 에이전트의 보상을 최대화하기

확률적 정책(학습 대상) :
$$
\pi(a|s)
$$  

정책을 함수로 표현하고 붙이는 변수 : 
$$
\theta
$$  

<small>
-정책을 학습하기 위해 수학적으로 표현  
-보통 신경망으로 $\theta$ 표현 _신경망을 도구로_  
-신경망의 가중치 W 매개변수 통칭  
<small>

신경망으로 구현한 정책 :
$$
\pi_\theta(a|s)
$$  

목적함수 J (objective function) :

$$
\tau
$$ 
는 정책 
$$
\pi_\theta(a|s)
$$ 를 따를때, 환경과의 상호작용으로 생성된 하나의 궤적 

$$
\tau = (S_0, A_0, R_0, S_1, A_1, R_1, \dots, S_T)
$$

→ 정책 
$$
\pi(a|s)
$$ 에 따른 행동의 결과로 얻은 시계열 데이터   

$$
G(\tau) = R_0 + \gamma R_1 + \gamma^2 R_2 + \cdots + \gamma^T R_T
$$

→  정책도 환경도 확률적이기 때문에 수익도 매번 달라질 수 있다.  
∴ 수익으로 정책이 좋은지를 판단해야하기 때문에 **수익의 기댓값이 목적함수**

- 정책으로 목적함수 설정 :  
정책을 사용해서 여러번 시도했을때 평균적으로 얼마나 보상을 받는지
**정책 경사법에서 목적함수J 는 경사상승법으로 최댓값을 찾는다.**  


$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]
$$


- 정책 갱신법

$$
θ ← θ+α∇_θJ(θ)
$$  


---


- 손실함수와 목적함수의 차이점

| 항목             | DQN (손실함수)                           | 정책경사법 (목적함수)                    |
|------------------|------------------------------------------|------------------------------------------|
| 학습 기준        | 오차(예측 vs 실제)                      | 기대 보상                               |
| 함수 이름        | Loss $$L(\theta)$$                      | Objective $$J(\theta)$$                 |
| 수식 예          | $$L(\theta) = (y - Q(s, a; \theta))^2$$ | $$J(\theta) = \mathbb{E}[G(\tau)]$$     |
| 최적화 방향      | 경사 하강 ↓                              | 경사 상승 ↑                              |
| 역할             | Q 추정 정확도 ↑                          | 정책 성능 ↑                              |

선형회귀는 손실함수L 은 Q값의 예측이 실제값과 얼마나 차이가 나는지를 구하고 이를 경사하강법으로 최소화 

---

[도출 과정]  

① 목적함수

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]
$$

② 기댓값의 미분 계산 - 기울기 구하기  

$$
\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\tau}[G(\tau)]
$$  

<small> 
- 기울기를 구하는 이유   
:기울기는 최대 증가 방향을 알려주는 벡터값이기 때문
<small> 

③ 실제 계산

$$
\nabla_\theta J(\theta)
= \mathbb{E}_{\tau} \left[ \sum_{t=0}^T G(\tau) \nabla_\theta \log \pi_\theta(A_t | S_t) \right]
$$

궤적을 여러번 샘플링해서 평균을 내는 것 (몬테카를로 샘플링 기법) 으로 정리  
**목적함수는 수익값 x 정책을 미분한 값의 평균**  

- 확률 그 자체보다 로그확률의 미분이 더 안정적이기 때문에 정책에 로그를 취한다  
- 정책 $$ 
\pi_\theta(a|s) 
$$ 를 θ로 미분하게 되면 경사가 구해진다. 
- $$
\nabla_\theta \log \pi_\theta(a|s)
$$ 는 정책이 그 행동을 선택하도록 만든 기여도의 벡터값을 뜻한다. 
- 수익값은 가중치의 역할을 한다 == 보상을 많이 받은 행동의 확률이 높인다.  

---

##### ⑶ 정책경사법 알고리즘 

- 구현 코드 1. 신경망 생성 (확률적 정책함수 정의)

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)          # 첫 번째 계층
        self.l2 = L.Linear(action_size)  # 두 번째 계층

    def forward(self, x):
        x = F.relu(self.l1(x))     # 첫 번째 계층에서는 ReLU 함수 사용
        x = F.softmax(self.l2(x))  # 두 번째 계층에서는 소프트맥스 함수 사용
        return x
```
- 복잡한 환경에서 확률적인 정책을 출력하기 위해 신경망 사용
- 이때 완전연결층이 1층이면 표현력이 약하고 3층이상이면 과적합 위험이 있기 때문에 안정적인 2층의 완전 연결 모델 구현 (카트폴 구현을 위한) 
- ReLU로 비선형성을 도입   
- Softmax 함수는 여러개의 값을 받고 전체합이 1이 되는 확률분포로 변환하는 역할  
ex) x = [2.1, 1.0] → softmax(x) = [0.75, 0.25]

```
입력: state (예: CartPole에서는 4차원 벡터)
↓
1층: Linear(4 → 128), ReLU
↓
2층: Linear(128 → 행동 수), Softmax
↓
출력: [0.75, 0.25] → 확률 분포 (정책 πθ(a|s))
```

- 구현코드 2. 에이전트 행동 & 학습

```python
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = [] #에피소드 데이터를 저장할 메모리
        self.pi = Policy(self.action_size) # 정책 신경망 생성
        self.optimizer = optimizers.Adam(self.lr) # 옵티마이저 설정 -> 파라미터를 업데이트해주는 알고리즘
        self.optimizer.setup(self.pi) # 정책 파라미터 등록 

    def get_action(self, state):
        state = state[np.newaxis, :]  # 차원 맞추기
        probs = self.pi(state)        # 순전파 수행, 정책신경망에 상태를 입력하여 확률을 출력
        probs = probs[0] # 확률을 1차원으로 정리
        action = np.random.choice(len(probs), p=probs.data)  # 확률적 행동 선택
        return action, probs[action]  # 선택된 행동과 확률 반환

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data) #정책 업데이트시 필요한 보상과 행동확률 저장

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):  # 수익 G 계산
            G = reward + self.gamma * G

        for reward, prob in self.memory:  # 손실 함수 계산
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []  # 메모리 초기화
```

- 목적함수가 아닌 손실함수를 쓰는 이유  

딥러닝은 기본적으로 backward(), step() 등의 메소드가 손실함수를 최소화하는 방향으로 파라미터를 업데이트하도록 이미 만들어져있다.  

목적함수는 스칼라값이며 -손실함수

| θ    | Lθ | -Lθ |
|-------------|-------------|--------------|
| θ_1 | 10          | -10          |
| θ_2 | 2           | -2           |

따라서 **'-손실함수'를 최소화하면 목적함수가 최대화된다**
loss += -F.log(prob) * G  


- 구현코드 3. 카트폴 실행

```python
episodes = 3000
env = gym.make('CartPole-v0', render_mode='rgb_array')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)  # 행동 선택
        next_state, reward, terminated, truncated, info = env.step(action)  # 행동 수행
        done = terminated | truncated

        agent.add(reward, prob)  # 보상과 행동의 확률을 에이전트에 추가
        state = next_state       # 상태 전이
        total_reward += reward   # 보상 총합 계산

    agent.update()  # 정책 갱신

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

from common.utils import plot_total_reward
plot_total_reward(reward_history)
```

- 에피소드별 보상 합계 추이

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/9_1.png" alt="1" width="500">
  <img src="/assets/images/9_2.png" alt="2" width="500">
</div>

--- 

[정책경사법 (Policy Gradient)]  
├── REINFORCE (Monte Carlo 기반)  
├── Actor-Critic  
│   ├── Advantage Actor-Critic (A2C)  
│   └── A3C, PPO 등...  


### 2. REINFORCE 알고리즘

앞선 정책경사법의 코드에서는 가중치 G를 모든 보상을 더한 값으로 계산했지만, 특정시간 t에서 행동 A이 좋은지 나쁜지는 행동 A를 하고 난 후의 보상의 총합으로 평가되기 때문에 행동 전에 얻은 보상은 관련이 없는 노이즈가 된다. REINFORCE 알고리즘은 이를 개선한 기법    
( **RE**ward **I**ncrement = **N**onnegative **F**ator X **O**ffset **R**einforcement X **C**haracteristic **E**ligibility )

$$
G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots + \gamma^{T - t} R_T
$$

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta (A_t \mid S_t) \right]
$$


```python
# 정책경사법

    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):  # 수익 G 계산
            G = reward + self.gamma * G

        for reward, prob in self.memory:  # 손실 함수 계산
            loss += -F.log(prob) * G

        loss.backward()
        self.optimizer.update()
        self.memory = []  # 메모리 초기화

```

```python
# REINFORCE 기법
    def update(self):
        self.pi.cleargrads()

        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G  # 수익 G 계산
            loss += -F.log(prob) * G     # 손실 함수 계산

        loss.backward()
        self.optimizer.update()
        self.memory = []

```
- 에피소드별 보상 합계 추이

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/9_3.png" alt="1" width="500">
  <img src="/assets/images/9_4.png" alt="2" width="500">
</div>

---

### 3. BASELINE 알고리즘

REINFORCE 알고리즘 개선 _ 분산을 줄이자  

⑴ 아이디어

현재의 결과만으로 비교를 하면 분산이 커지는 문제점이 있다.  
실제 결과 - 과거의 평균(예측값)으로 도출되는 차이로 비교를 하게 되면 분산이 작아진다. 

⑵ 적용  

REINFORCE는 G_t가 클수록 정책을 강화하기때문에 모든 행동이 강화된다. 

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} G_t \nabla_\theta \log \pi_\theta(A_t | S_t) \right]
$$


하지만 **BASELINE은 G_t가 평균보다 높으면 정책을 강화하고 낮으면 약화하는 방식으로 분산을 줄여, 학습을 안정화시킬 수 있다.  

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} (G_t - b(S_t)) \nabla_\theta \log \pi_\theta(A_t | S_t) \right]
$$

b(S_t) 를 베이스라인이라고 부르며 이는 임의의 함수로 보통 가치함수를 사용하기 때문에 b(S_t)= 𝑉𝜋(𝑆_𝑡)  
하지만 정책경사법에서는 정책이 계속 달라지기 때문에 정확한 가치함수의 계산이 불가능하기 때문에 신경망을 통해 근사  
→ 다시 가치 기반 기법(Value-based)으로 접근  
→ 베이스라인은 가치기반과 정책기반의 혼합된 형태  

### 4. 행위자와 비평자

지금까지의 정책경사법 코드에서는 MC법을 사용해서 에피소드가 다 끝난 후에 업데이트를 진행했다. 
MC는 에피소드가 종료되지 않으면 학습을 할 수 없고, 데이터를 다 모아야 학습이 가능하기 때문에 실시간 시스템이나 계속해서 움직이는 로봇 등의 학습에 한계가 있다.  
하지만 TD법을 사용하면 한스텝마다 업데이트를 진행할 수 있게 되면서 무한한 에피소드 환경에서도 학습이 가능해지고 빠르고 안정적인 학습이 가능하다.  

MC법 ) 가치함수 V_t -> G_t  
TD법 ) 가치함수 V_t -> R_t + γ * V_(t+1)  

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \left( R_t + \gamma V_w(S_{t+1}) - V_w(S_t) \right) \nabla_\theta \log \pi_\theta(A_t \mid S_t) \right]
$$

⇒ TD-기반 행위자-비평자(Actor-Critics)  

어떤 행동을 할지 결정하는 정책을 행위자 Actor로 보고 이 정책을 평가하는 가치함수를 비평자 Critic로 나누어보기 때문에 
이때 정책과 가치함수는 모두 신경망으로 두 신경망을 병렬로 학습시켜 V의 값이 R_t + γ * V_(t+1) 에 근사하도록 학습시킨다  

- 코드 구현  

```python
if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class PolicyNet(Model):  # 정책 신경망
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)  # 확률 출력
        return x


class ValueNet(Model):  # 가치 함수 신경망
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)

    def get_action(self, state):
        state = state[np.newaxis, :]  # 배치 처리용 축 추가
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]  # 선택된 행동과 해당 행동의 확률 반환

    def update(self, state, action_prob, reward, next_state, done):
        # 배치 처리용 축 추가
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # 가치 함수(self.v)의 손실 계산
        target = reward + self.gamma * self.v(next_state) * (1 - done)  # TD 목표
        target.unchain()
        v = self.v(state)  # 현재 상태의 가치 함수
        loss_v = F.mean_squared_error(v, target)  # 두 값의 평균 제곱 오차

        # 정책(self.pi)의 손실 계산
        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        # 신경망 학습
        self.v.cleargrads()
        self.pi.cleargrads()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()


episodes = 3000
env = gym.make('CartPole-v0', render_mode='rgb_array')
agent = Agent()
reward_history = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action, prob = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        agent.update(state, prob, reward, next_state, done)

        state = next_state
        total_reward += reward

    reward_history.append(total_reward)
    if episode % 100 == 0:
        print("episode :{}, total reward : {:.1f}".format(episode, total_reward))

from common.utils import plot_total_reward
plot_total_reward(reward_history)
```

### 5. 정리  

정책 경사법은 가치기반기법의 한계를 보완한 기법 

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \Phi_t \nabla_\theta \log \pi_\theta(A_t | S_t) \right]
$$

Φ_𝑡​는 각 기법마다 달라지는 보상의 척도이자 기여도 계수  

1. $$
\Phi_t = G(\tau) $$          # 가장 단순한 정책 경사법  

2. $$
\Phi_t = G_t$$           # REINFORCE  

3. $$
\Phi_t = G_t - b(S_t) 
$$      # 베이스라인을 적용한 REINFORCE  

4. $$
\Phi_t = R_t + \gamma V(S_{t+1}) - V(S_t) 
$$ # 행위자–비평자 (Actor–Critic)  
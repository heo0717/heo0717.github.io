---
layout: archive
title: " Chapter.5 Monte Carlo Method "
permalink: /coding/deeplearning/MC/ 
author_profile: true
math: true
---

Index

1. 몬테카를로법이란?
    - 모델표현방식
    - 증분방식
2. 몬테카를로법 _ 정책평가
3. 몬테카를로법 _ 정책제어(개선)
    - MC 와 DP의 차이점
    - Epsilon ? <개선1>
    - Epsilon-Greedy ?
    - 고정값 α
4. 온/오프 정책
    - 온-정책
    - 오프-정책
    - 중요도 샘플링



## Monte Carlo Method

### 1. 몬테카를로법이란?

이전에 동적프로그래밍을 사용하여 최적가치함수와 최적 정책을 찾을 수 있었지만, DP는 반드시 환경 모델이 필요한 모델 기반이라는 한계가 존재  
모델기반이 왜 한계냐, 실제 환경에서는 모델을 알 수 없는 상황이 많고 상태 전이가 복잡한 요인으로 결정되기 때문에 s' = f(s|a) 를 명확히 알기가 어렵기 때문  
따라서 에이전트를 실제로 움직이며 데이터를 수집해야하고 
몬테카를로법은 모델환경을 모르는 경우에 반복적인 샘플링(실험)을 통해 데이터를 모으고 가치함수를 측정할 수 있다
- 실험: 상태(State), 행동(Action), 보상(Reward)

---
warming up..

#### 모델표현방식

1. 분포 모델 ( distribution model )
- _모든_ 상태 전이 확률을 명확히 알고 있음
- 수학식으로 상태 전이와 보상을 표현 가능함

2. 샘플 모델 ( sample model )
- _일부_ 확률적 샘플 데이터만 사용하여 추정
- 무한히 샘플링하는 경우, 분포 모델에 수렴하는 특징
- 환경 모델 없이 경험만으로 평가 가능

∴ 몬테카를로법은 샘플 모델 방식 사용

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/MonteCarlo1.png" alt="MC2" width="500">
</div>

- 구현 (1)

```python
trial = 1000
V, n = 0, 0

for _ in range(trial):
    s = sample()
    n += 1
    V += (s - V) / n
    print(V)
```

샘플이 많으면 많을수록, 안정적이어진다 = 분산이 작아진다

- 증분 방식 계산  
어떤 값의 평균을 반복적으로 업데이트할 때, 평균을 즉시 업데이트 하는 방법  
모든 데이터를 저장할 필요가 없음  

[n번째까지의 평균]
general expression ) V_n = ( s_1 + s_2 + s_3 + … + s_n ) / n

incremental update ) V_n = V_(n-1) + 1/n(s_n - V_(n-1))  

->  new_estimate = old_estimate + a * ( reward - old_estimate )

---

  
### 2. 몬테카를로 정책 평가 Monte Carlo Policy Evaluation

1) 가치 함수 V 구하기

$$
v_π(s) = E_π[G|s]
$$

- G는 미래에 받을 보상의 합 (Return)
- 몬테카를로법에서는 G를 실험 데이터로 추정
    따라서 강화학습에서 실험데이터를 얻을때 시작상태(위치)를 변경 가능

    ⇒ V_π(s) = ( G(1) + G(2) + G(3) + … + G(n) ) / n

    (할인율은 가정함)
    
- 계산 최적화 방법

    move : A -> B -> C

$$
    G_a = R_0 + r * R_1 + r ** 2 * R_2
    G_b = R_1 + r * R_2
    G_c = R_2

    ⇒

        G_a = R_0 + r G_b
        G_b = R_1 + r G_c
        G_c = R_2
$$

∴ 순서를 뒤집어, C -> B -> A 순서로 계산을 하면 반복 계산을 피할 수 있다.   

그렇다면 효율적인 계산은 모델 상태가 몇개인지는 알아야할 수 있는 것인가 ?  
:대부분 몬테카를로법이 사용하는 환경은 유한한 상태 공간을 가정.
상태의 가짓수를 모르면 탐색이 불가능해지기 때문.  
== 아래 코드에서 action size가 필요한 이유

```python
class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
```

self.pi = 정책 설정  
self.V = 가치함수  
self.memory = 행동으로 얻은 ' s , a , r ' 저장  
self.cnts = 증분계산방식을 이용한 가치함수 
np.random.choice(actions, p=probs) = 확률분포에 따라 하나씩 샘플링 

Episode :  start -> ...... -> end ( arrived goal state )
에피소드가 종료될때 마다 에이전트는 reverse order로 G값을 계산하여 V 업데이트

```python

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):  # 역방향으로(reserved) 따라가기
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]
```

data = (state, action, reward) 튜플 형태

$$
S^0 , A^0, R^0, S^1 , A^1, R^1, .... Time series data
$$

save as

$$
(S^0,A^0,R^0), (S^1,A^1,R^1) ..
$$

목표지점에서의 보상은 항상 0 이기 때문에 마지막 데이터는 저장하지 않음

---

```python
env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):  # 에피소드 1000번 수행
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)             # 행동 선택
        next_state, reward, done = env.step(action)  # 행동 수행

        agent.add(state, action, reward)  # (상태, 행동, 보상) 저장
        if done:   # 목표에 도달 시
            agent.eval()  # 몬테카를로법으로 가치 함수 갱신
            break         # 다음 에피소드 시작

        state = next_state

# 가치함수 시각화
env.render_v(agent.V)
```

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/MonteCarlo2.png" alt="MC2" width="500">
</div>

---

  
### 3. 몬테카를로 정책 제어 Monte Carlo Policy Improvement

$$
\mu(s) = \arg\max_a Q(s, a)
$$

강화학습에서는 환경모델을 모르기 때문에 **Q함수** 를 사용  
정책평가를 통해 Q값을 개선하고 개선된 Q값에 따라 정책 업데이트 반복

- DP와 MC의 차이점

| 구분               | DP 방식                                    | MC 방식                                      |
|--------------------|---------------------------------------------|----------------------------------------------|
| 필요한 정보    | 환경 모델 필요 (P, R)                      | 모델 불필요, 경험 데이터만 필요              |
| 정책 평가 방법 | 벨만 기대 방정식으로 반복 계산             | 에피소드 결과(G)를 평균해서 계산            |
| 업데이트 시점  | 각 상태마다 반복적으로                    | 에피소드 종료 후 한 번에                    |
| 적용 가능 환경 | 모델 기반 환경 (model-based)              | 모델 없이도 가능 (model-free)               |
| Q 또는 V 계산  | 수식 기반                                  | 샘플 평균 기반                               |

- DP
$$
V(s) = sum_a π(a|s) * sum_s' P(s'|s,a) * [R(s,a,s') + γV(s')]
$$

- MC
$$
V(s) ← average of returns G from episodes starting at s
$$

- 구현(3)

```python
class RandomAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):  # 역방향으로(reserved) 따라가기
            state, action, reward = data
            G = self.gamma * G + reward
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]


env = GridWorld()
agent = RandomAgent()

episodes = 1000
for episode in range(episodes):  # 에피소드 1000번 수행
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)             # 행동 선택
        next_state, reward, done = env.step(action)  # 행동 수행

        agent.add(state, action, reward)  # (상태, 행동, 보상) 저장
        if done:   # 목표에 도달 시
            agent.eval()  # 몬테카를로법으로 가치 함수 갱신
            break         # 다음 에피소드 시작

        state = next_state


env.render_v(agent.V)
```

- 구현(4)

```python
def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs
```

- Epsilon의 의미

ε 은 작은 확률을 나타내는 그리스 문자로, 수학이나 머신러닝 분야에서 _조금만 탐색한다_ 라는 의미로 사용  

    - ε=0 ; 탐욕적 정책
    - ε=1 ; 무작위 정책

- Epsilon-Greedy 정책

최적 행동을 따르되, 일정확률 무작위 행동을 시도하여 오차를 줄임  
남은 확률 ( 1 - ε )으로 Q값이 가장 높은 행동을 선택 (활용)
ε 확률로 무작위 행동을 선택 (탐색)

- 고정값 alpha 

증분방식계산식)  V_n+1 = V_n + 1/(n+1) * (s_(n+1) - V_(n)) 
코드적용 ver.) Q(s,a) = Q(s,a) + 1/N(s,a) * (G - Q(s,a))  
  
N(s,a)는 상태-행동의 횟수  
각 상태마다 행동의 빈도가 다를 수 있기 때문에 상태 s와 행동 a를 같이 저장하며, Q값을 추정하기 위해서는 상태와 행동을 구분해야함  
 에피소드가 진행될수록 N이 커져서 점점 업데이트의 값(최신값)이 줄어드는 방식이 됨  
 
 gamma == 1 / N(s,a) / 할인율의 역할

 - 할인율 ( discount factor , γ )  
 미래 보상에 대한 현재 가치를 결정하는 값 / 미래보상에 대한 가치를 줄여나가는 비율
 γ가 1에 가까울수록 미래의 보상을 더 중요하게 생각하고, 0에 가까울수록 미래의 보상을 덜 중요하게 생각.  
 _( <-> 학습률 : Q값을 업데이트할 때 새로운 정보가 얼마나 반영될지 비율을 조절하는 역할)_

 ∴ 현재 증분계산방식으로는 초기에 잘못된 방식으로 수렴했을때, 에피소드가 점점 증가하면서 업데이트의 폭이 계속 작아져서 바뀌지않는 학습정체가 발생하고, 잘못된 방향으로 흘러갈 수 있음 

 - 고정값 (alpha, α)  
 증분계산방식으로 생기는 오차를 해결하기 위해 **고정된 학습률(NOT 할인율)**을 도입
 학습률을 도입하게 되면 각 데이터에 대한 가중치가 기하급수적으로 커지는 **지수이동평균**
 지수이동평균은 최신 데이터일수록 가중치를 크게 반영

 ━ G값은 반복적인 정책개선을 하면서 에피소드마다 정책이 달라지게 되어 확률분포가 비정상. 이러한 비정상문제에는 지수이동평균이 적합  

[Final Code]
```python
class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1  # (첫 번째 개선) ε-탐욕 정책의 ε
        self.alpha = 0.1    # (두 번째 개선) Q 함수 갱신 시의 고정값 α # exponential moving average
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0) # Q instead V
        # self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs) #exploration

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            # self.cnts[key] += 1
            # self.Q[key] += (G - self.Q[key]) / self.cnts[key]
            self.Q[key] += (G - self.Q[key]) * self.alpha
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)
```
- update functional structure

    1. G = reward + self.gamma * G <보상은 현재의 가치가 가장 중요하기때문에 할인율 적용>
    - accumulate from the last G
    2. alpha <Q함수는 업데이트 반영 정도를 조절하기 위해 학습률 적용>
    - exponential moving average
    3. self.Q[...]
    - update Q(s,a) based on experimental return G
    4. self.pi[state] = greedy_probs(...)
    - Q was changed in step 3. so, policy would have to updated


```python
env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.add(state, action, reward)
        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q)
```

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/MonteCarlo4.png" alt="MC4" width="500">
</div>

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/MonteCarlo3.png" alt="MC3" width="500">
</div>

---  

### 온/오프-정책

- 온-정책 on-policy

    스스로 쌓은 경험을 토대로 자신의 정책을 개선  
    **ε-탐욕 정책(ε-greedy)을 사용하여 탐색과 최적 행동을 동시에 진행**  

- 오프-정책 off-policy

    다른 사람이 실행하는 정책으로 얻은 데이터를 바탕으로 학습. 즉, 내가 직접 하지 않은것도 학습에 반영  
    이때, 행동 정책(행동으로 탐색한 정책 Behavior policy)과 대상 정책(학습하려는 정책 Target policy)이 다를 수 있음. | 행동정책: ε-greedy 또는 완전 랜덤, 대상정책: greedy
    **탐색 방식: Behavior Policy ≠ Target Policy**

    - 내 행동이어도 무작위탐색을 하게 되면 의도치 않은 데이터가 생기게 되고, 실제로 다른 에이전트의 경험을 재사용하기도 함.

- 중요도 샘플링 Importance sampling

∴ 오프-정책인 경우, 행동정책에서 얻은 샘플 데이터의 값은 학습 보정이 필요하기 때문에 중요도 샘플링 활용  
중요도 샘플링은 어떤 확률분포의 기댓값을 다른 확률 분포에서 샘플링한 데이터를 사용하여 계산하는 기법  


― 구하고자 하는 것: 정책 π를 따른 기댓값  

$$
\mathbb{E}_\pi[x] = \sum_x \pi(x) \cdot x
$$

― 실제로 모은/얻은 데이터: 정책 b로 행동

따라서, 
$$
 \rho(x^{(i)}) = \frac{\pi(x^{(i)})}{b(x^{(i)})}
$$


정책 π와 정책 b의 차이 보정을 위한 **중요도 비율(ρ, rho)** 을 곱한다  

$$
\mathbb{E}_\pi[x] = \sum_x \frac{\pi(x)}{b(x)} \cdot x
$$


― 그러나 보정이 커지면 분산이 커지게 되어 학습이 불안정해짐  
→ 두 분포를 비슷하게 조정  
    행동 정책을 완전히 무작위가 아닌 정책 π 에 비슷하도록 정책 b를 조정 or ε-greedy  
    **그러나 최적화하려는 것은 대상정책 π 이기때문에 정책 b를 π 에 가깝게 탐험해야 효율적인 학습**

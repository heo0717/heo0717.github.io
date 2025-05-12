---
layout: archive
title: " chapter6. TD "
permalink: /coding/deeplearning/TD/ 
author_profile: true
math: true
---

## TD

Index  

1. TD 학습이란
    - MC와 TD의 차이점
2. SARSA
    - On-policy
    - Off-policy
3. Q-Learning
    - 벨만방정식과 SARSA
    - 벨만최정방정식과 Q-Learning
    - 분포모델 / 샘플모델
4. Simple Q-Learning


### 1. TD학습이란

강화학습에서 가치함수를 업데이트할때 에피소드가 끝날때까지 기다리지 않고 **즉시 업데이트**를 진행하는 방법  
MC법은 에피소드가 종료되어야 G값을 리턴하지만 TD는 다음상태로 전이되는 순간 보상을 기반으로 업데이트  

- Return

general)
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$
  
TD)
$$
G_t = R_{t+1} + \gamma G_{t+1}  
$$  
미래의 전체 Return G 대신 다음 상태의 가치추정치를 사용 (Bootstraping)

- value function

MC)
$$
V_\pi(S_t) \leftarrow V_\pi(S_t) + \alpha (G_t - V_\pi(S_t))
$$

TD)
$$
V_\pi(S_t) \leftarrow V_\pi(S_t) + \alpha \left(R_{t+1} + \gamma V_\pi(S_{t+1}) - V_\pi(S_t)\right)
$$

- MC와 TD의 차이점  

    MC target : G_t  
    MC의 목표는 에피소드가 종료된 후의 리턴값인 G_t를 사용  
    즉, 실제로 모든 상태와 보상을 경험하고 난 후 값을 업데이트하기 때문에 _분산이 큰 편_   
    ex) 자율주행 / 도로 상황이 매우 다양하기 때문에 에피소드 종료후 결과를 얻음  

    TD target : 
    $$
    R_t + \gamma V(S_{t+1})
    $$
    TD 방법은 즉시 업데이트를 하기 때문에 에피소드가 종료될 필요가 없어 _빠르지만 약간의 오차가 발생한다._ 오차는는 업데이트를 많이 할수록 줄어들어 최적화된다.


- 구현(1)
```python
class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]  # 목표 지점의 가치 함수는 0
        target = reward + self.gamma * next_V
        self.V[state] += (target - self.V[state]) * self.alpha
```
---

### SARSA 

- On-policy

SARSA 는 TD학습의 확장으로 Q값을 업데이트하는 방식  
온-정책 학습으로, 에이어전트가 실제 선택한 행동에 대해서만 학습  
TD는 V를 구하기 위함 / SARSA는 A와 Q를 구하기 위함

$$
S_t, A_t, R_(t+1), S_(t+1), A_(t+1)
$$

한스텝 뒤(next-state)를 반영한다는 점이 TD와의 차이  
S → A → R → S' → A' 순서로 학습 진행  
(각 상태에서 각행을 선택할때는 ε-greedy 정책 사용)

Q함수 대상, TD법 V 갱신식 / SARSA UPDATE)  
$$
Q_\pi(S_t, A_t) \leftarrow Q_\pi(S_t, A_t) + \alpha \left(R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1}) - Q_\pi(S_t, A_t)\right)
$$

- 구현(2)  

```python

import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict, deque
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs


class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)  # deque 사용

    def get_action(self, state):
        action_probs = self.pi[state]  # pi에서 선택
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q = 0 if done else self.Q[next_state, next_action]  # 다음 Q 함수

        # TD법으로 self.Q 갱신
        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        
        # 정책 개선
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = SarsaAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)  # 매번 호출

        if done:
            # 목표에 도달했을 때도 호출
            agent.update(next_state, None, None, None)
            break
        state = next_state

env.render_q(agent.Q)

```

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/sarsa_2.png" alt="MC2" width="500">
  <img src="/assets/images/sarsa_1.png" alt="MC2" width="500">
</div>

- Off-policy

SARSA는 원래 On-policy 학습 방식이지만 Off-policy 도 있음  
행동정책과 학습정책이 동일하다면 충분한 경험이 반영되지 못할 수 있기 때문  

$$
\rho = \frac{\pi(A_{t+1} \mid S_{t+1})}{b(A_{t+1} \mid S_{t+1})}
$$

$$
Q_\pi(S_t, A_t) \leftarrow Q_\pi(S_t, A_t) + \alpha \cdot \rho \cdot \left[R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1}) - Q_\pi(S_t, A_t)\right]
$$

- 구현(3)  

```python

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict, deque
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs


class SarsaOffPolicyAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)

    def get_action(self, state):
        action_probs = self.b[state]  # 행동 정책에서 가져옴
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 1
        else:
            next_q = self.Q[next_state, next_action]
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]  # 가중치 rho 계산

        # rho로 TD 목표 보정
        target = rho * (reward + self.gamma * next_q)
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 각각의 정책 개선
        self.pi[state] = greedy_probs(self.Q, state, 0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = SarsaOffPolicyAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)

        if done:
            agent.update(next_state, None, None, None)
            break
        state = next_state

# [그림 6-9] 오프-정책 SARSA로 얻은 결과
env.render_q(agent.Q)
```

---

### Q-Learning

off-policy SARSA는 중요도 샘플링을 이용해야한다.  하지만 중요도 샘플링은 결과가 불안정할 수 있다. 행동 정책과 대상 정책의 확률 분포가 다를수록 중요도 샘플링의 가중치도 변동성이 커질 수 있다.  
이러한 문제를 해결하는 것이 Q러닝  
따라서 Q러닝은 TD법의 off-policy인데 중요도 샘플링을 사용하지 않는 방식이다.

- 벨만방정식과 SARSA

- 벨만최정방정식과 Q-Learning

Q함수가 MAX 연산자로 행동 A를 택하기 때문에 중요도 샘플링을 이용한 보정이 필요가 없다  

```python
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld
from common.utils import greedy_probs


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)  # 행동 정책
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.b[state]  # 행동 정책에서 가져옴
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:  # 목표에 도달
            next_q_max = 0
        else:     # 그 외에는 다음 상태에서 Q 함수의 최댓값 계산
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        # Q 함수 갱신
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 행동 정책과 대상 정책 갱신
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)


env = GridWorld()
agent = QLearningAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

#  Q 러닝으로 얻은 Q 함수와 정책
env.render_q(agent.Q)
```

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/Q1.png" alt="Q1" width="500">
  <img src="/assets/images/Q2.png" alt="Q2" width="500">
</div>

- 분포모델 / 샘플모델

1. 분포모델

```python
class RandomAgent:
    def __init__(Self):
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi =defaultdict(lambda: random_actions)

    def get_actions(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs= list(action_probs.values())
        return np.random.choice(actions, p=probs)
```

각 상태에서의 행동 확률 분포를 self.pi 변수에 유지  
확률 분포를 명시적으로 유지하는 것이 분포모델의 특징  

2. 샘플모델

```python
class RandomAgent:
    def get_action(self,state):
        return np.random.choice(4)
```

샘플모델은 확률분포를 유지할 필요가 없기 때문에 단순히 네가지 행동 중 하나를 무작위로 선택하도록 간단히 구현 가능  

---

### simple Q-learning

```python
import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for importing the parent dirs
from collections import defaultdict
import numpy as np
from common.gridworld import GridWorld


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:  # epsilon의 확률로 무작위 행동
            return np.random.choice(self.action_size)
        else:                                # (1 - epsilon)의 확률로 탐욕 행동
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha


env = GridWorld()
agent = QLearningAgent()

episodes = 1000
for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, next_state, done)
        if done:
            break
        state = next_state

env.render_q(agent.Q)
```

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/Q_simple1.png" alt="Q_1" width="500">
  <img src="/assets/images/Q_simple2.png" alt="Q_2" width="500">
</div>
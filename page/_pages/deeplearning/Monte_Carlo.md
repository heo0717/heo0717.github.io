---
layout: archive
title: " Monte Carlo Method "
permalink: /coding/deeplearning/Monte/ 
author_profile: true
math: true
---

## Monte Carlo Method

### what is Monte Carlo Method

before, using DP we can find optimum value function &  optimum policy. but it has limit that certainly knowing the environmental model.

But in real world, there are many situation that we can’t know. So, now we need to move the agent.

Monte Carlo Method is estimating way that samples the data repeatedly, and based on the results. In reinforcement learning, we can use Monte Carlo Method for assuming the value function by experiment

experiment means state & acting & reward

* basement

Grid world in DP part, it cleared  about the state(location) and reward data. put another way, we can using p(s’ | a, s) and r(s, a, s’). and also if the state transition is crucial, we able to express in an expression that s’ = f(s,a)

But actually, the state transition is determined by complex factors, not conclusive. in addition to, if it is determined by probability, we might know the state transition theoretically but there will be a lot of calculations.

* the ways expressing model

- distribution model : **entire** probability distribution
- sample model : definite **(partial)** sample data / if doing infinitely repetition of sampling, that distribution will be same with probability distribution. 

and it is easier than distribution model because it just can change parameter.

⇒  Monte Carlo Method is the way calculating expected value in sample model way. (  doing a lot of sampling and getting the expected value / and the result will true in accordance with the law of large numbers. )

( the model using in DP, is distribution model )

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/MonteCarlo1" alt="MC2" width="500">
</div>

- realization (1)

```python

trial = 1000
V, n = 0, 0

for _ in range(trial):
    s = sample()
    n += 1
    V += (s - V) / n
    print(V)

```

the more sample, the more reliable it is. and it can also be said that the variance becomes smaller.
(variance =  deviation from the correct answer)

+ ) incremental update ← calculation method

general expression ) V_n = ( s_1 + s_2 + s_3 + … + s_n ) / n

incremental update ) V_n = V_(n-1) + 1/n(s_n - V_(n-1))
/ new_estimate = old_estimate + a * ( reward - old_estimate ) /

when  update repeatedly some value’s average, it is a method of updating the average immediately with just one new value without storing all of the data.

---

### how to policy evaluate using Monte Carlo Method

1. find the value function

value function ) v_π(s) = E_π[G|s]
(G is return that inquired reward with discount rate)
    
    
    In Monte Carlo Method, actual G is sample data and we need to gather like that data.
    
    ⇒ V_π(s) = ( G(1) + G(2) + G(3) + …. + G(n) ) / n
    
    ( the discount rate is assumed )
    
    if the policy(π) or the state transition , either  is probabilistic, the reward must be different probability.
    
    and  that time, we can use Monte Carlo Method.
    
    →  value function in entire state, can find repeatedly changing the start state  / so, in reinforce learning, we can setting start location(state) contrary to reality.

- an efficient method of calculation

    if state transition move A → B → C, we can getting 3 data about each reward at A and B, C _state just one sample.
    
    G_a = R_0 + r * R_1 + r ** 2 * R_2
    G_b = R_1 + r * R_2
    G_c = R_2

    it means,

    G_a = R_0 + r G_b
    G_b = R_1 + r G_c
    G_c = R_2

    Thus, 
    if we calculate C → B → A ( reverse order ) , can avoid duplicate calculations.

- realization (2)
according to _probability distribution_

step method

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

why action_size is needed
1. To loop over all possible actions
2. To create a uniform random policy - each action should be assigned an equal probability
3. To select a random action from valid ones -?

self.pi = make policy
self.V = value function
self.memory = experinence gained by action _ ' s , a , r '
self.cnts = G's average using incremental update
np.random.choice(actions, p=probs) = sampling the action one by one according to the probability distribution


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
data = (state, action, reward) <- tuple
    self.memory.append(data)

$$
S^0 , A^0, R^0, S^1 , A^1, R^1, .... Time series data
$$

save as

$$
(S^0,A^0,R^0), (S^1,A^1,R^1) ..
$$

_but the last one is not saved because V at the goal always 0_

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
  <img src="/assets/images/MonteCarlo2" alt="MC2" width="500">
</div>

---

## how to policy control

policy evaluation <-> policy improvement

$$
\mu(s) = \arg\max_a Q(s, a)
$$

Generally, in reinforce learning, we should use Q function because we don't know the env model
and If we improve Q function, we should evaluate them

* difference DP and MC

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

- realization(3)

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
- realization(4)

if there is no exploration, return policy will be inaccurate

```
def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}  #{0: ε/4, 1: ε/4, 2: ε/4, 3: ε/4}
    action_probs[max_action] += (1 - epsilon)
    return action_probs
```

- epsilon = small
    ε is simply a Greek letter commonly used to represent a small probability.
    In mathematics and machine learning, ε is traditionally used to express the idea of "exploring just a little."

    When ε → 0, the policy becomes fully greedy (no exploration).

    When ε → 1, the policy becomes completely random (full exploration).

- epsilon-greedy 
    : follow the optimal action
    : tried in a different direction

    Use the remaining (1 − ε) to choose the action with the highest Q-value (exploitation)

```
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

    1. G = reward + self.gamma * G 
    - accumulate from the last G
    2. alpha = 1 / cnt
    - incremental mean 
    3. self.Q[...]
    - update Q(s,a) based on experimental return G
    4. self.pi[state] = greedy_probs(...)
    - Q was changed in step 3. so, policy would have to updated

episode = start -> ...... -> end ( arrived goal state )


```
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

# [그림 5-17] 및 [그림 5-18]
env.render_q(agent.Q)
```

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/MonteCarlo4" alt="MC4" width="500">
</div>

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/MonteCarlo3" alt="MC3" width="500">
</div>

---

### importance sampling

어떤 확률 분포의 기댓값을 다른 확률분포에서 샘플링한 데이터를 사용해 계산하는 기법

기댓값 수식 
$$
\mathbb{E}_\pi[x] = \sum_x \pi(x) \cdot x
$$

기댓값 보정 수식
$$
\mathbb{E}_\pi[x] = \sum_x \frac{\pi(x)}{b(x)} \cdot x
$$

- off-policy

어떤 정책(행동)으로 데이터를 모았는데, 평가하려는 정책과 다를 때 문제 발생
- 데이터는 ε-greedy 를 사용해서 데이터 편향을 줄임. 학습시에 사용
- 평가는 geedy _평가를 랜덤으로 할 필요가 없기 때문에. 정책 평가 최적화 단계

샘플링:  
$$
x^{(i)} \sim b \quad (i = 1, 2, \dots, n)
$$

중요도 샘플링으로 근사한 기대값:

$$
\mathbb{E}_\pi[x] \approx \frac{1}{n} \sum_{i=1}^n \rho(x^{(i)}) \cdot x^{(i)}
$$

(여기서 중요도 비율:  
$$
\rho(x^{(i)}) = \frac{\pi(x^{(i)})}{b(x^{(i)})}
$$
)
---
layout: archive
title: " Dynamic Programming "
permalink: /coding/deeplearning/dp/ 
author_profile: true
math: true
---

## 1. 벨만 방정식 - 벨만 최적 방정식

- Action-Value Function

<small>벨만방정식</small>
<br>
$$
Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a') \right]
$$

<small>벨만최적방정식</small>
<br>
$$
Q_*(s, a) = \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma \max_{a'} Q_*(s', a') \right]
$$

- State-Value Funtion

<small>벨만방정식</small>
<br>
$$
v^\pi(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v^\pi(s') \right]
$$

<small>벨만최적방정식</small>
<br>
$$
v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right]
$$

<small>최적정책</small>

$$
μ_*(s) = argmax_a * q_*(s,a)
$$

# 2. 선형 - 비선형

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/linear.png" alt="선형" width="500">
  <img src="/assets/images/non-linear.png" alt="비선형" width="500">
</div>

벨만최적방정식에서는 max가 사용되기때문에 구간에 따라 기울기나 형태가 다른 비선형  
따라서 구간별로 반복적인 계산이 요구됨


# 3. DP

<div style="text-align: left;">
$$
π(a|s) > Q(s,a) >  V_\pi(s) > μ
$$
</div>

- 반복정책평가  

  정책평가 : 정책 π가 주어졌을때 그 정책의 가치함수 V(s), Q(s,a)를 구하는 문제  
  정책제어 : 정책을 조정해서 최적 정책을 만들어내는 것

  pi : 정책 (확률값)  
  V : 가치함수  
  env : 환경  
  gamma : 할인율  

  
- 갱신 bellman update  

  계속 업데이트하며 정책을 평가해야하기때문에 반복적인 계산이 필요  
  <br>

$$
V_k+1(s) = \sum_a \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_k(s') \right]
$$ 


   Q함수를 통해 도출한 정책은 결정적 정책이기 때문에 가치함수를 사용한 정책평가와 최종 최적 정책도 답이 하나로 수렴  
   <small>( softmax나 greedy알고리즘의 경우 확률적 정책을 활용하여 탐색 )</small>

```python

#갱신양을 정하는 경우 - 정적 / (정확도 < 속도) 인경우

V = {'L1': 0.0, 'L2': 0.0}
new_V = V.copy()
gamma = 0.9
theta = 0.0001
cnt = 0  # 반복 횟수 기록

for _ in range(100):
    new_V['L1'] = 0.5 * (-1 + gamma * V['L1']) + 0.5 * (1 + gamma * V['L2'])
    new_V['L2'] = 0.5 * (0 + gamma * V['L1']) + 0.5 * (-1 + gamma * V['L2'])

    delta_L1 = abs(new_V['L1'] - V['L1'])
    delta_L2 = abs(new_V['L2'] - V['L2'])
    delta = max(delta_L1, delta_L2)

    V = new_V.copy()
    print(V)

```

```python

# 오차가 작아질때 - 동적
# 덮어쓰기 방식

V = {'L1': 0.0, 'L2': 0.0}

gamma = 0.9
theta = 0.0001
cnt = 0  # 반복 횟수 기록

while True:
    t = 0.5 * (-1 + gamma * V['L1']) + 0.5 * (1 + gamma * V['L2'])
    delta_L1 = abs(t - V['L1'])
    V['L1'] = t

    t = 0.5 * (0 + gamma * V['L1']) + 0.5 * (-1 + gamma * V['L2'])
    delta_L1 = max(delta, abs(t - V['L2']))
    V['L2'] = t

    cnt += 1
    if delta < theta:
        print(V)
        print('갱신횟수:', cnt)
        break

```
-  Setting __ GridWorld

```python

#grid_world

import numpy as np

class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
        self.action_meaning = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.reward_map = np.array([
            [0, 0, 0, 1],
            [0, None, 0, -1],
            [0, 0, 0, 0]
        ])
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self): return len(self.reward_map)

    @property
    def width(self): return len(self.reward_map[0])

    @property
    def shape(self): return self.reward_map.shape

    def actions(self): return self.action_space

    def states(self):
        for h in range(self.height): #0,1,2
            for w in range(self.width): #0,1,2,3
                yield (h, w) #L->R , Up -> Bottom

    #action
    
    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT
        move = action_move_map[action]
        ny, nx = state[0] + move[0], state[1] + move[1]

        # 격자 밖이거나 벽이면 원래 위치 유지
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            return state
        if (ny, nx) == self.wall_state:
            return state

        return (ny, nx)

    def reward(self, state, action, next_state):
        return self.reward_map[next_state]    

```

```python

#visualization

    def render_v(self, V=None):

        matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows용
        matplotlib.rcParams['axes.unicode_minus'] = False     # 마이너스(-) 기호 깨짐 방지

        data = np.zeros(self.shape)
        for y in range(self.height):
            for x in range(self.width):
                if (y, x) == self.wall_state:
                    data[y, x] = np.nan
                elif V and (y, x) in V:
                    data[y, x] = V[(y, x)]
                else:
                    data[y, x] = 0

        plt.imshow(data, cmap='bone', interpolation='nearest')
        for y in range(self.height):
            for x in range(self.width):
                if (y, x) == self.wall_state:
                    continue
                plt.text(x, y, f'{data[y, x]:.2f}', va='center', ha='center', fontsize=9)
        # plt.title("Grid World 상태 가치 함수 시각화")
        plt.colorbar()
        plt.show()

```
- Test

```python

#test_random V

from grid_world import GridWorld
import numpy as np

env = GridWorld()
V = {}

# 상태마다 랜덤한 가치 할당 (시각화 테스트용)
for state in env.states():
    V[state] = np.random.uniform(-1, 1)

env.render_v(V)

```
<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/random2.png" alt="" width="800">
  <img src="/assets/images/random1.png" alt="" width="800">
</div>

## 정책평가법

모델환경 (gridworld)을 알기 때문에 에이전트를 직접 움직이지않고 기댓값만 계산

```python
from collections import defaultdict
from grid_world import GridWorld

#이동 - 업데이트
def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_probs = pi[state]
        new_v = 0
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action) #s -> s'
            r = env.reward(state, action, next_state)
            new_v += action_prob * (r + gamma * V[next_state]) #V힘수

        V[state] = new_v 

    return V

#업데이트 반복
def policy_eval(pi, V, env, gamma=0.9, threshold=0.001):
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)
        delta = 0 #변화량 check
        for state in V.keys():
            delta = max(delta, abs(V[state] - old_V[state]))
        if delta < threshold:
            break
    return V
```

```python
env = GridWorld()
# 무작위 정책
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

# 초기 가치 함수
V = defaultdict(lambda: 0)

# 정책 평가
V = policy_eval(pi, V, env, 0.9)

# 결과 시각화
env.render_v(V)
```
상태전이가 결정적인 경우의 상태가치함수  
<small> 탐색하지않는 순수 최적화 알고리즘이기 때문에 결정적 정책 

$$
 V(s) = \sum_a \pi(a|s) \cdot \left[ r(s, a, s') + \gamma V(s') \right]
$$

### V(s')은 사실 이전의 값

defaultdict로 V_0 = 0 초기화

V(s) = r + gamma * V_0

초기 Q table  

| 상태   | ↑ (0) | ↓ (1) | ← (2) | → (3) |
|:-----:|:------:|:------:|:------:|:------:|
| (0,0) |  0.0   |  0.0   |  0.0   |  0.0   |
| (0,1) |  0.0   |  0.0   |  0.0   |  0.0   |
| (0,2) |  0.0   |  0.0   |  0.0   |  0.0   |
| (0,3) |  0.0   |  0.0   |  0.0   |  0.0   |
| (1,0) |  0.0   |  0.0   |  0.0   |  0.0   |
| (1,1) |  0.0   |  0.0   |  0.0   |  0.0   |
| (1,2) |  0.0   |  0.0   |  0.0   |  0.0   |
| (1,3) |  0.0   |  0.0   |  0.0   |  0.0   |
| (2,0) |  0.0   |  0.0   |  0.0   |  0.0   |
| (2,1) |  0.0   |  0.0   |  0.0   |  0.0   |
| (2,2) |  0.0   |  0.0   |  0.0   |  0.0   |
| (2,3) |  0.0   |  0.0   |  0.0   |  0.0   |

1회차 Q table 

| 상태   | ↑ (0) | ↓ (1) | ← (2) | → (3) |
|--------|-------|-------|-------|-------|
| (0, 0) | 0.00  | 0.00  | 0.00  | 0.00  |
| (0, 1) | 0.00  | 0.00  | 0.00  | 0.90  |
| (0, 2) | 0.90  | 0.81  | 0.00  | 1.00  |
| (0, 3) | 1.00  | 1.00  | 1.00  | 1.00  |
| (1, 0) | 0.00  | 0.00  | 0.00  | 0.00  |
| (1, 2) | 0.90  | 0.73  | 0.81  | -0.10 |
| (1, 3) | 1.00  | 0.66  | 0.81  | -0.10 |
| (2, 0) | 0.00  | 0.00  | 0.00  | 0.00  |
| (2, 1) | 0.00  | 0.00  | 0.00  | 0.73  |
| (2, 2) | 0.81  | 0.73  | 0.00  | 0.66  |
| (2, 3) | -0.10 | 0.66  | 0.73  | 0.66  |

1회차 V 값

['0.00', '0.00', '0.25', '0.00']        
['0.00', '0.00', '-0.19', '-0.04']      
['0.00', '0.00', '-0.04', '-0.27'] 

2회차 Q table

| 상태   | ↑ (0) | ↓ (1) | ← (2) | → (3) |
|--------|-------|-------|-------|-------|
| (0, 0) | 0.00  | 0.00  | 0.00  | 0.81  |
| (0, 1) | 0.81  | 0.81  | 0.00  | 0.90  |
| (0, 2) | 0.90  | 0.81  | 0.81  | 1.00  |
| (0, 3) | 1.00  | 1.00  | 1.00  | 1.00  |
| (1, 0) | 0.00  | 0.00  | 0.00  | 0.00  |
| (1, 2) | 0.90  | 0.73  | 0.81  | -0.10 |
| (1, 3) | 1.00  | 0.66  | 0.81  | -0.10 |
| (2, 0) | 0.00  | 0.00  | 0.00  | 0.66  |
| (2, 1) | 0.66  | 0.66  | 0.00  | 0.73  |
| (2, 2) | 0.81  | 0.73  | 0.66  | 0.66  |
| (2, 3) | -0.10 | 0.66  | 0.73  | 0.66  |

2회차 V 값

['0.00', '0.06', '0.28', '0.00']        
['0.00', '0.00', '-0.25', '-0.13']      
['0.00', '-0.01', '-0.13', '-0.43']  

처음 V함수를 구할때 V[next_State]는 사실상 defaultdict에 의해서 0  
따라서 V(s') 는 사실상 미래의 가치함수가 아니라 이전에 계산한 가치함수값

실제로 미래 값을 알 수 없고, 그  미래 값이 예측하고자하는 부분이기 때문에 알고 있는 과거의 데이터를 이용하는 것
< 벨만 방정식은 단지 공식일뿐 >

## 정책반복법

- Greedy policy

```python
def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
    return max_key
```
```python
def greedy_policy(V, env, gamma):
    pi = {}

    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state] #q함수
            action_values[action] = value

        max_action = argmax(action_values) 
        #가장 보상이 큰 하나의 행동 key  반환
        action_probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        action_probs[max_action] = 1.0 #결정적정책
        pi[state] = action_probs

    return pi #최적의 정책 반환
```
action_values = {}  
action_values[action] = value   
ex. action_values = {0: -0.5, 1: -0.3, 2: -0.8, 3: -0.1}  
현재상태에서 가능한 모든행동에 대한 기대가치 저장

```python

def policy_iter(env, gamma=0.9, threshold=0.001, is_render=True):
    
    pi = defaultdict(lambda: {a: 0.25 for a in env.actions()})  # uniform 초기 정책
    V = defaultdict(lambda: 0)
    cnt = 0

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)
        pi_new = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi_new)

        if pi_new == pi:
            print(f"총 반복횟수: {cnt}회")
            break

        pi = pi_new
        cnt += 1

    return pi, V
```

[정책평가 set]  
eval_onestep() : V함수 계산  
\+ Policy_eval() : 수렴할때까지 V함수 계산

[정책개선 set]   
greedy_policy() : 업데이트된 V함수 값으로 정책 계산 후 정책 업데이트  
\+ policy_iter() : 정책이 수렴할때까지 계산 (전체루프)


- 평가와 개선반복

GPI : 평가와 개선을 반복해서 개선해나아가는 전체적인 틀

만약 100번을 실행할때 100번째에 다른 행동에 대한 보상이 더 높게 나와서 답이 틀려버린다면?
=> 정책이 수렴하지않은 것. 

- 수렴할 수 밖에 없는 이유
1. 행동의 가짓수와 공간이 유한하면 수렴  
2. 할인율이 1보다 작기때문에 결국엔 수렴
3. $$V_π′(s) ≥ V_π(s)$$  
V는 argmax를 사용하기 때문에 항상 더 개선됨   

- V함수와 Q함수의 반복  

Q(s, a) = r + γ * V(s')
(=) value = r + gamma * V[next_state]

## 가치반복법

```python
from collections import defaultdict
from common.gridworld import GridWorld
from ch04.policy_iter import greedy_policy

def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        if state == env.wall_state:
            continue

        action_values = []
        for action in env.actions():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            value = reward + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)  # V(s) ← max_a Q(s,a)
    return V

def value_iter(V, env, gamma, threshold=0.001, is_render=False):
    while True:
        if is_render:
            env.render_v(V)

        old_V = V.copy()
        V = value_iter_onestep(V, env, gamma)

        delta = max(abs(V[s] - old_V[s]) for s in V.keys())
        if delta < threshold:
            break
    return V

#-------------------------------------------------------------------------

if __name__ == "__main__":
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter(V, env, gamma)
    pi = greedy_policy(V, env, gamma) #마지막 한번만 처리
    env.render_v(V)
    env.render_pi(pi)
```

V[state] = max(reward + gamma * V[next_state])  
이 한줄로 정책반복법에서의 평가와 개선으로 나뉜 두 부분을 한번에 처리

<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <img src="/assets/images/policy_eval1.png" alt="" width="800">
</div>
<div style="display: flex; gap: 10px; margin-bottom: 30px;">
  <figure>
    <img src="/assets/images/dp_arr_1.png" alt="정책반복법 예시" width="300">
    <figcaption style="text-align: center;">1</figcaption>
  </figure>
  <figure>
    <img src="/assets/images/dp_arr_100.png" alt="가치반복법 예시" width="300">
    <figcaption style="text-align: center;">100</figcaption>
  </figure>
</div>

## 4. 정리

- 정책반복법_평가와 개선 과정: 
$$
  \pi(s|a) → V → Q → μ_0 → V → Q → μ_1 ...
$$
모든 상태의 Q를 계산 후 한번에 V를 업데이트하는 동기방식  
[정책 반복법] 100회 실행 소요 시간: 0.0050초

- 가치반복법_평가와 개선 과정: 
$$
  \pi(s|a) → V → V → ... → μ
$$
각 상태의 V를 즉시 업데이트하는 비동기방식  
사용하는 v(s')의 업데이트 시점이 가치반복법에서 더 빠르기 때문에 훨씬 빠른 속도  
[가치 반복법] 100회 실행 소요 시간: 0.0020초

    _현실에서는 완벽한 정책이 아닌 괜찮은 정책이 필요_

- DP

1. 반복적인 계산 수행
2. 한번 계산한 값을 중복 수행하지 않음
3. 주로 최대, 최소값을 통해 값을 구하는 문제
4. 모델환경을 알고 있는 경우에 사용
---
layout: archive
title: " Chapter.8 DQN"
permalink: /coding/deeplearning/DPN/ 
author_profile: true
math: true
---

## DQN  

Index  
1. Open AI Gym  
2. DQN의 핵심
    - 경험재생
    - 목표신경망
3. DQN과 아타리
    - 아타리
    - CNN 
4. DQN 확장
    - Double DQN
    - 우선순위 경험재생 PER
    - Dueling DQN 
5. 정리

---

- DQN 
DQN(Deep Q Network)은 Q러닝과 신경망을 이용한 기법으로 _경험재생_ 과 _목표신경망_ 기술이 더해진 것이다.  

### 1. Open AI Gym  

- 기초지식 

이번장부터는 그리드월드를 벗어나 상태가 더 다양한 환경으로 확장한다. 이때 OpenAI Gym라이브러리가 제공하는 _카트폴_ 을 사용한다.  

env = gym.make('CartPole-v0', render_mode='human') #화면 출력  
env = gym.make('CartPole-v0', render_mode='rgb_array') #화면 출력 X  

(카트폴은 카트를 움직여 막대의 균형을 잡는 균형잡기게임)

---

```python
import gym

env = gym.make('CartPole-v0', render_mode='human')

state = env.reset()[0]
print('상태:', state)

action_sapce = env.action_space
print('행동의 차원 수:', action_space)
# 출력결과
# 상태: [ 0.03453657 -0.01361909  -0.02143636 0.02152179 ]
# 행동으 차원 수: Discrete(2)
```

- state는 원소가 4개인 배열  
    카트의 위치   카트의 속도  막대의 각도  막대의 각속도 (회전속도)  

- Discrete으로 출력되는 이유 
    action space가 이산형으로 두가지 선택지가 있다는 뜻  
    0 : 왼쪽 / 1 : 오른쪽

```python
action = 0 #or 1
next_state, reward, terminated, truncated, info = env.step(action)
```

실제로 행동을 하게 되면, 결과로 5가지를 얻을 수 있다. 
1. next_state : 다음상태
2. reward : 보상 ( float type _ 카트폴에서는 균형이 유지되는 동안 보상은 항상 1 )
3. terminated : 목표 상태 도달 여부
    카트폴에서는 막대각도가 12도를 초과하거나 카트가 화면을 벗어날때 True 반환
4. truncated : MDP 범위 밖의 종료 조건 충족 여부
    행동 200회를 초과할때 True 반환환
5. info : 추가 정보

```python
import numpy as np
import gym

env = gym.make('CartPole-v0', render_mode='human')
state = env.reset()[0]
done = False

while not done:  # 에피소드가 끝날 때까지 반복
    env.render()  # 진행 과정 시각화
    action = np.random.choice([0, 1])  # 행동 선택(무작위)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated      # 둘 중 하나만 True면 에피소드 종료
env.close()
```
---

### 2. DQN의 핵심 

⑴ 신경망의 단점_ 과대적합 overfitting

딥러닝의 신경망은 복잡한 관계를 학습하고 예측할 수 있기 때문에 표현력이 높다고 표현한다. 따라서 입력 데이터의 패턴을 잘 포착한다는 장점이 있지만, 표현력이 너무 높으면 학습 데이터에 과하게 맞춰질 수 있는 **과대적합** 이 발생해서 실제로 새로운 상황에는 대응하기 어려운 문제가 발생한다. 

**DQN = Q-learning + 신경망**  

DQN은 신경망의 학습을 안정화하기 위해 **경험재생** 과 **목표 신경망** 기술을 사용한다

⑵ 경험재생 

- 지도학습과 Q러닝의 차이점 1 [학습데이터]

□ 지도학습

훈련용 데이터셋에서 일부 무작위 추출 - 미니배치(mini-batch)  
미니배치를 만들때는 데이터가 편향되지 않도록 주의 / 그래서 무작위 추출  
미니배치를 사용하여 신경망의 매개변수 갱신  
  
□ Q러닝  

에이전트가 환경에서 어떤 행동을 취할때마다 데이터를 생성하고, 시간 t에서 얻은 E_t = (S_t, A_t, R_t, S_(t+1)) 로 Q함수 갱신  
이때 **E_t는 경험데이터** 경험데이터는 시간의 흐름에 따라 얻을 수 있고, 경험데이터간에는 상관관계가 존재한다.  

_따라서 Q러닝은 편향된 데이터를 학습한다는 점에서 지도학습과 차이점이 있다._

- 경험 재생 experience reply

따라서 이러한 학습데이터의 차이를 메우기 위한 기법이 경험재생이다. 

1. 에이전트가 경험한 데이터를 **버퍼** 에 저장한다. (일시적으로 데이터를 보관하는 저장소 )  
2. Q함수를 갱신할때는 버퍼로부터 경험데이터를 무작위로 추출하여 (미니배치) 사용  

이는 경험데이터간의 상관관계를 약화시켜 데이터의 편향을 줄일 수 있고, 경험 데이터를 반복해서 사용할 수 있기 때문에 데이터 효율이 높아진다.  

_**경험재생은 오프정책 한정으로 사용된다.** 온정책은 현재 정책에서 얻은 데이터만을 사용하기 때문에 과거에 수집한 데이터는 사용할 수 없다._ 

- 구현 1

```python
from collections import deque
import random
import numpy as np
import gym


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self): 
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.int32)
        return state, action, reward, next_state, done


env = gym.make('CartPole-v0', render_mode='human')
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):  # 에피소드 10회 수행
    state = env.reset()[0]
    done = False

    while not done:
        action = 0  # 항상 0번째 행동만 수행
        next_state, reward, terminated, truncated, info = env.step(action) # 경험 데이터 획득
        done = terminated | truncated

        replay_buffer.add(state, action, reward, next_state, done)   # 버퍼에 추가
        state = next_state

# 경험 데이터 버퍼로부터 미니배치 생성
state, action, reward, next_state, done = replay_buffer.get_batch()
print(state.shape)       # (32, 4)
print(action.shape)      # (32,)
print(reward.shape)      # (32,)
print(next_state.shape)  # (32, 4)
print(done.shape)        # (32,)
```

---

⑶ 목표신경망 

- 지도학습과 Q러닝의 차이점 2 [정답레이블]

□ 지도학습

지도학습은 모델이 무엇을 예측해야할지 정답이 주어진 상태에서 학습을 한다. → 정답레이블이 부여된다.   
  
□ Q러닝  

Q러닝에서는 'TD목표'가 지도학습의 정답레이블과 같은 역할이지만 TD목표는 Q함수가 갱신될때마다 달라지게 된다. 
  
TD 목표 = 보상 + 미래예측값  
  
- 목표신경망 target network  

- 구현 step (1)  

TD목표값을 고정하기 위한 방식으로, QNet 클래스를 이용하여 에이전트인 DQNAgent 클래스가 원본신경망 qnet과 구조가 같은 신경망 qnet_target을 갖도록 준비한 후, sync_qnet 메소드를 통해 두 신경망을 동기화한다. 이때 **deepcopy**를 사용해 모든 데이터를 완벽히 복제한다.  
TD 목표가 계속바뀌지않고, 일정주기마다 TD목표인 qnet_target을 업데이트하는 방식  

```python
import copy
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


class QNet(Model):  # 신경망 클래스
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:  # 에이전트 클래스
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000  # 경험 재생 버퍼 크기
        self.batch_size = 32      # 미니배치 크기
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)         # 원본 신경망
        self.qnet_target = QNet(self.action_size)  # 목표 신경망
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)            # 옵티마이저에 qnet 등록

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]  # 배치 처리용 차원 추가
            qs = self.qnet(state)
            return qs.data.argmax()

    def sync_qnet(self):  # 두 신경망 동기화
        self.qnet_target = copy.deepcopy(self.qnet)
```

- 구현 step (2)  

update가 호출되면 버퍼에 경험데이터를 추가하고, 버퍼에서 미니배치로 데이터를 추출한다.  
미니배치의 사이즈는 관습적으로 2의 제곱수를 사용하며, 현재 코드에서는 32를 사용한다. 카트폴에서 하나의 상태는 [막대위치, 카트속도, 막대각도, 각속도] 4가지로 구성되어 있기때문에 state.shape = (32,4) 이다. 

```python
    def update(self, state, action, reward, next_state, done):
        # 경험 재생 버퍼에 경험 데이터 추가
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return  # 데이터가 미니배치 크기만큼 쌓이지 않았다면 여기서 끝

        # 미니배치 크기 이상이 쌓이면 미니배치 생성
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(axis=1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q

        loss = F.mean_squared_error(q, target)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()
```

- q = qs[np.arange(self.batch_size), action] 코드의 기능

에이전트가 실제 수행한 행동에 해당하는 Q값만 골라내기 위한 목적  

qs[i]에는 i번째 상태에서 왼쪽을 선택한 경우와 오른쪽을 선택한 경우, 각 행동의 Q값이 저장되어 있다. 그 중에서 실제로 한 행동의 Q값을 추출한다.  

- (1-done) mask

target = reward + (1 - done) * gamma * next_q  

done은 에피소드가 끝났는지를 나타내는 boolean 값   
-done = 1 : 에피소드 끝  
-done = 0 : 에피소드 계속  

에피소드가 끝났을때는 마지막 보상이 0이기때문에 next_q를 사용하지 않는데, 1-done을 마스크로 사용함으로써 에피소드가 끝났을때 0이 곱해져서 미래의 Q값을 target에 적용하지 않기 위한 방법 

- 구현 step (3) _ 실행

```python
episodes = 300      # 에피소드 수
sync_interval = 20  # 신경망 동기화 주기(20번째 에피소드마다 동기화)
env = gym.make('CartPole-v0', render_mode='rgb_array')
agent = DQNAgent()
reward_history = [] # 에피소드별 보상 기록

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    if episode % 10 == 0:
        print("episode :{}, total reward : {}".format(episode, total_reward))


# [그림 8-8] 「카트 폴」에서 에피소드별 보상 총합의 추이
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.plot(range(len(reward_history)), reward_history)
plt.show()


# 학습이 끝난 에이전트에 탐욕 행동을 선택하도록 하여 플레이
agent.epsilon = 0  # 탐욕 정책(무작위로 행동할 확률 ε을 0로 설정)
state = env.reset()[0]
done = False
total_reward = 0

while not done:
    action = agent.get_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated | truncated
    state = next_state
    total_reward += reward
    env.render()
print('Total Reward:', total_reward)
```

- 하이퍼파라미터 hyperparameter

사람이 미리 설정한 값 

할인율 (gamma) = 0.98  
학습률 (lr) = 0.0005  
ε-탐욕 확률 (epsilon) = 0.05  
경험 재생 버퍼 크기 (buffer_size) = 100000  
미니배치 크기 (batch_size) = 32  
에피소드 수 (episodes) = 300  
타겟 동기화 주기 (sync_interval) = 20  
신경망 구조 (계층 수, Linear 계층의 노드 수 등)

---

### 3. DQN과 아타리

⑴ 아타리 Atari  

OpenAI Gym_ 아타리 게임용 환경 제공 <game | Pong>  
Pong은 공을 주고 받는 top뷰의 탁구경기   

강화학습의 이론은 MDP를 전제로 진행. MDP는 최적의 행동을 선택할때 필요한 정보가 **현재 상태**에 담겨 있다. 하지만 Pong에서는 공의 진행방향을 모르는 상태로는 최적 행동을 알아낼 수 없기때문에 **부분관찰 마르코프 결정과정 POMDP**를 사용한다.  

- POMDP  

Pong과 같은 비디오게임의 형태에서는 하나의 이미지가 아닌 네개의 프레임이 연속된 이미지를 겹쳐 하나의 상태로 취급한다. 이 방법을 통해 상태의 전이 (공의 움직임)을 알 수 있게된다.  

POMDP는 MDP와 달리 현재상태 s를 완전히 관찰할 수 있는 것이 아니라 부분적으로만 관찰이 가능하다. 
즉, MDP처럼 현재만을 관찰하는게 아니라 **과거의 관찰을 고려하여 행동을 결정하는 것**이 POMDP이다.  


- 전처리 

프레임 중첩을 위한 전처리 작업이 필요  

1. 이미지 주변 불필요한 요소 제거
2. 그레이 스케일
3. 이미지 크기 조정
4. 정규화 ( 이미지 원솟값 0 ~ 1사이로 )

⑵ CNN

CNN은 합성곱 신경망 Convoluional Neutral Network 으로, 이미지나 시계열 데이터 등 공간적 패턴을 갖는 데이터를 잘다루는 신경망 구조이다.  

카트폴 게임은 완전연결계층으로 구성된 신경망을 이용했으나 아타리처럼 이미지 데이터를 다룰 때는 CNN 즉, **합성곱 계층** 을 이용한 신경망을 사용한다. 

- 합성곱계층 
입력된 이미지에 필터를 슬라이딩하면서 특징을 추출하는 계층. 
가장자리, 선, 모서리, 패턴을 자동으로 감지 

---

### 4. DQN 확장

DQN은 딥러닝에서 가장 유명한 알고리즘

- DQN의 한계

1. Q값의 추정이 과대평가
2. 모든 경험을 똑같이 학습
3. 모든 행동에 대해 똑같은 방식으로 Q값 추정

$$
Q(s, a) = r + \gamma \max_{a'} Q(s', a')
$$


⑴ Double DQN

$$
Q(s, a) = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_{\theta}(s', a'))
$$

⑵ PER

$$
\delta = r + \gamma \max Q(s', a') - Q(s, a)
$$

⑶ Dueling DQN

$$
Q(s, a) = V(s) + A(s, a)
$$

| 비교 항목         | Double DQN                     | PER                             | Dueling DQN                         |
|------------------|----------------------------------|----------------------------------|--------------------------------------|
| 목적              | Q값 과대평가 방지              | 중요한 경험을 우선적으로 학습  | Q값을 가치와 이점으로 분리         |
| 무엇을 다르게?    | Q값 계산 방식 분리             | 경험 샘플 선택 확률 다르게 함  | Q함수 구조 자체를 바꿈             |
| 수학적 포인트     | max → argmax 분리              | TD오차 기반 샘플링               | \( Q = V + A \)                      |
| 효과              | 학습 안정성 증가               | 빠르고 효율적인 수렴             | 정밀한 Q값 근사, 빠른 학습 가능     |


### 5. 정리

- Q러닝의 문제점
    1. 학습데이터의 편향
    2. 학습목표가 계속 바뀜

- DQN의 해결책
    1. 경험재생 : 순차적데이터를 무작위로 섞어서 사용
    2. 목표신경망 : Q(s',a)를 고정된 네트워크로 계산. 

- 왜?
    Q러닝을 딥러닝으로 구현하려면 불안정하다. 지도학습은 가장 안정적이며 효율적인 학습방식이기 때문에 Q러닝에서 그 안정성을 흉내내기 위한 해결책  
    결국, **DQN은 지도학습의 안정성 + 강화학습의 스스로 학습하는 장점을 절충한 학습방법** 
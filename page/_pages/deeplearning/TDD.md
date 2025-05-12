---
layout: archive
title: " Monte Carlo Method "
permalink: /coding/deeplearning/TDD/ 
author_profile: true
math: true
---

## TD

Monte Carlo Method can be used in a one-off task ( with an end )
but if theres's an end to the task ( an ongoing task? )

In MonteCarlo Method, V can be updated when the episode is over becuz the G will be confirmed at that time.

TD ( temporal Difference ) is _immediately update and imporve policy_ whenever the action is over, not wating the end of episode

- Return

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$


- value function

$$
V_\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right]
$$

---

- updating value function using MC Method

$$
V_\pi(S_t) \leftarrow V_\pi(S_t) + \alpha \left( G_t - V_\pi(S_t) \right)
$$

- calculating expectation of value function in DP 

$$
V_\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_\pi(s') \right]
$$

based on Bellman's equation, updating V in series order

- updating value function in TD

$$
v_\pi(s) = \sum_{a, s'} \pi(a \mid s) \, p(s' \mid s, a) \left[ r(s, a, s') + \gamma v_\pi(s') \right]
$$

$$
v_\pi(s) = \mathbb{E}_\pi \left[ R_t + \gamma v_\pi(S_{t+1}) \mid S_t = s \right]
$$

$$
V_\pi(S_t) \leftarrow V_\pi(S_t) + \alpha \left( R_{t+1} + \gamma V_\pi(S_{t+1}) - V_\pi(S_t) \right)
$$

### comparison between MD and TD

MD's target : G_t
mostly, MD has bigger variance ( ex. autonomous driving )

TD's target : R_t + gamma * V(S_(t+1))
updating V by using V => bootstraping
bias has exhist because it is esimation. but the more update the more accurate (<-> MD )

- realization(5)

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

---

### SARSA

sarsa method is included in on-policy

State, Action, Reward, next State, next Action

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]
$$


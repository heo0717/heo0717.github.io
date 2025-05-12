---
layout: single
title: " Bellman's equation "
permalink: /coding/deeplearning/bellman/
author_profile: true
math: true
---

마르코프결정과정에서 환경과 행동이 확률적이고 지속가능한 과제인 경우 사용

# 확률과 기댓값

# 상태가치함수

가장 많은 보상을 받을 수 있는 행동을 선택하기 위한 함수

조건: 상태 s , 정책 π

# 행동가치함수

조건: 상태 s , 정책 π, 행동 a

# 벨만최적방정식

$$

v_*(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_*(s') \right]

$$
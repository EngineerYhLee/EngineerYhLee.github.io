---
published: true
title: "손실함수(Loss Function)"
description: "머신러닝 모델에서 손실함수의 역할과 다양한 종류에 대한 설명"
header:
  teaser: /assets/images/Loss_Function.png
  og_image: /assets/images/Loss_Function.png
  image_description: "손실함수(Loss Function)"
date: 2024-10-01
last_modified_at: 2024-10-01
toc: true
toc_sticky: true
use_math: true
categories:
  - BasicConcepts
tags:
  - AIFundamentals
---
## 손실함수(Loss Function)란 무엇인가?
손실함수(Loss Function)는 머신러닝과 딥러닝에서 매우 중요한 개념으로, 모델이 예측한 출력값과 실제 정답값 간의 차이를 측정하는 데 사용됩니다. 손실함수의 값은 모델이 얼마나 잘못 예측했는지를 나타내며, 이 값이 작을수록 모델의 예측이 정확하다는 것을 의미합니다.

머신러닝의 목표는 모델이 예측을 할 때 이 손실함수의 값을 최소화하는 것이며, 이를 통해 더 정확한 모델을 만들 수 있습니다. 손실함수는 크게 **회귀 문제**와 **분류 문제**에서 각각 다르게 정의됩니다.

### 손실함수 수식 예시

$$
L(y, \hat{y}) = f(y - \hat{y})
$$

여기서 $L$은 손실 함수, $y$는 실제 값, $\hat{y}$는 예측 값입니다.

---

## 손실함수의 종류

### 1. 평균 제곱 오차(Mean Squared Error, MSE)

**MSE**는 회귀 문제에서 자주 사용되는 손실 함수입니다. 이 함수는 예측값과 실제값의 차이를 제곱하여 평균을 취합니다.

수식은 다음과 같습니다:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- $n$: 데이터 포인트의 수
- $y_i$: 실제값
- $\hat{y}_i$: 예측값

MSE는 오차가 클수록 더 큰 패널티를 부여하므로, 큰 오차를 줄이는 데 효과적입니다. 하지만 이상치(outlier)에 민감하다는 단점이 있습니다.

### 2. 평균 절대 오차(Mean Absolute Error, MAE)

**MAE**는 MSE와 비슷하지만, 차이를 제곱하지 않고 절대값을 사용합니다. MAE는 이상치에 덜 민감하다는 특징이 있습니다.

수식은 다음과 같습니다:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

### 3. 교차 엔트로피(Cross-Entropy)

**교차 엔트로피(Cross-Entropy)**는 주로 분류 문제에서 사용되며, 확률 분포 간의 차이를 측정하는 데 사용됩니다. 이 함수는 예측된 확률 분포와 실제 분포 사이의 거리(엔트로피)를 계산합니다.

이진 분류 문제의 경우 교차 엔트로피는 다음과 같이 계산됩니다:

$$
CE = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

- $y_i$: 실제 클래스 (0 또는 1)
- $\hat{y}_i$: 예측 확률

### 4. 후버 손실(Huber Loss)

**후버 손실(Huber Loss)**는 MSE와 MAE의 장점을 결합한 손실 함수로, 이상치에 덜 민감하면서도 큰 오차를 줄이는 데 효과적입니다. 이 함수는 작은 오차에 대해서는 MSE처럼 작동하고, 큰 오차에 대해서는 MAE처럼 작동합니다.

수식은 다음과 같습니다:

$$
L_{\delta}(a) =
\begin{cases}
\frac{1}{2} a^2 & \text{if } |a| \leq \delta \\
\delta (|a| - \frac{1}{2} \delta) & \text{if } |a| > \delta
\end{cases}
$$

여기서 $\delta$는 변화를 조절하는 임계값입니다.

---

## 손실함수의 역할

손실함수는 모델이 예측을 개선하는 방향으로 학습하게 돕는 역할을 합니다. 손실값을 최소화하기 위해 **경사 하강법(Gradient Descent)**과 같은 최적화 알고리즘이 사용되며, 손실값을 줄이는 방향으로 가중치(weight)와 편향(bias)이 조정됩니다.

### 경사 하강법(Gradient Descent)과 손실함수의 관계

경사 하강법은 손실함수의 기울기를 계산하여 가중치를 업데이트하는 방식입니다. 이 기울기는 손실함수의 미분값으로 구할 수 있으며, 손실함수가 더 가파른 부분에서는 가중치 업데이트 폭이 커지고, 평평한 부분에서는 작아집니다.

수식 예시:

$$
w = w - \eta \cdot \frac{\partial L}{\partial w}
$$

여기서 $w$는 가중치, $\eta$는 학습률(learning rate), $L$은 손실 함수입니다.

---

## 손실함수 선택 방법

모델의 학습 목표와 데이터의 특성에 맞는 손실함수를 선택하는 것이 중요합니다. 손실함수 선택은 모델 성능에 직접적인 영향을 미치므로, 문제 유형에 맞는 함수를 선택하는 몇 가지 기준을 고려해야 합니다.

### 1. 문제 유형에 따른 선택

- **회귀 문제**: 연속적인 수치를 예측할 때는 `MSE`(평균 제곱 오차)나 `MAE`(평균 절대 오차)를 주로 사용합니다.
- **이진 분류 문제**: 예측 값이 두 가지 클래스 중 하나일 경우에는 `Binary Cross-Entropy`를 사용합니다.
- **다중 분류 문제**: 클래스가 여러 개인 경우에는 `Categorical Cross-Entropy` 또는 `Sparse Categorical Cross-Entropy`를 사용합니다.

### 2. 데이터 특성에 따른 선택

- **이상치가 많은 경우**: `MAE`는 이상치에 덜 민감하므로 적합할 수 있습니다. `Huber Loss`도 좋은 대안이 될 수 있습니다.
- **확률 분포를 다룰 때**: 분류 문제에서 확률적 출력을 원하면 `Cross-Entropy` 계열 손실함수를 사용해야 합니다.

### 3. 모델의 특성에 따른 선택

- **회귀 신경망**: 일반적으로 `MSE`를 많이 사용하지만, 가끔 `Huber Loss`가 더 적합할 수 있습니다.
- **GANs (Generative Adversarial Networks)**: 생성 모델과 판별 모델에 따라 각기 다른 손실함수가 사용됩니다. 

---

## 파이토치(Pytorch) 코드 예시

아래 코드는 파이토치를 이용한 손실함수 코드 예시입니다. 

```python
# 평균 제곱 오차(Mean Squared Error, MSE) 예시

import torch
import torch.nn as nn

# 예시 데이터
y_true = torch.tensor([1.0, 2.0, 3.0])
y_pred = torch.tensor([1.5, 2.5, 3.5])

# MSE 손실함수 정의
mse_loss = nn.MSELoss()

# 손실값 계산
loss = mse_loss(y_pred, y_true)
print(f'MSE Loss: {loss.item()}')
```

```shell
MSE Loss: 0.25
```
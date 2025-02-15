---
published: false
title: "(6) Fully Connected Layer"
description: "Fully Connected Layer의 이해와 응용"
header:
  teaser: /assets/images/fully_connected_layer.png
  og_image: /assets/images/fully_connected_layer.png
  image_description: "Fully Connected Layer"
date: 2025-02-15
last_modified_at: 2025-02-15
toc: true
toc_sticky: true
use_math: true
categories:
  - CNN
tags:
  - DeepLearning
  - CNN
  - Architecture
---

[//]: # (작성 양식)
[//]: # (# 마크다운: kramdown)
[//]: # (# 수식 예시 => 수식이 필요할 경우에만 작성:  $$y = f(z) = f\left(\sum_{i=1}^n w_i x_i + b\right)$$)
[//]: # (# 언어: 한글, 주요 단어는 영문과 같이 표시)
[//]: # (# 참고문헌: 실제 존재하는 논문들을 형식에 맞게 기재)

## 1. 인용 예시 
- **LeNet-5 등장**  
  얀 르쿤과 동료들은 LeNet-5와 같이 초기 CNN 모델에서 Fully Connected Layer를 활용하여 손글씨 숫자 인식(MNIST) 등의 실용 문제에서 탁월한 성능을 입증했습니다[^5].

# Fully Connected Layer의 이해와 응용

## 2. Fully Connected Layer의 기본 개념
**Fully Connected Layer (FC Layer)**는 모든 입력 뉴런이 모든 출력 뉴런과 연결된 계층으로, 입력 벡터 \(\mathbf{x}\)에 대해 선형 변환을 수행하여 출력 벡터 \(\mathbf{y}\)를 산출합니다. 수학적으로는 아래와 같이 표현됩니다:

$$
\mathbf{y} = W \mathbf{x} + \mathbf{b}
$$

여기서,  
- \(W\)는 학습되는 가중치 행렬 (크기: \(M \times N\))  
- \(\mathbf{b}\)는 편향 벡터  
- \(N\)은 입력 차원, \(M\)은 출력 뉴런 수를 의미합니다.

이 후, 보통 **활성화 함수**(예: ReLU, Sigmoid, Softmax)가 적용되어 비선형성을 부여합니다.

## 3. CNN에서 Fully Connected Layer의 역할
CNN에서는 컨볼루션 및 풀링 계층들을 통해 추출된 고차원 특징들을 1차원 벡터로 **Flatten**하여 FC Layer에 입력합니다. 이 계층은 추출된 특징을 종합하여 최종 분류 혹은 회귀를 수행하는 **분류기(Classifier)** 역할을 하며, 보통 출력층에서는 **Softmax** 함수를 통해 각 클래스에 대한 확률 분포를 산출합니다.

초기의 LeNet-5, AlexNet 등은 여러 개의 큰 FC Layer를 사용했지만, 최근에는 **Global Average Pooling**이나 **1×1 Convolution**을 활용하여 FC Layer의 파라미터 수를 크게 줄이는 추세입니다.

## 4. Fully Connected Layer가 학습과 연산량에 미치는 영향
FC Layer는 모든 입력과 출력 사이의 연결로 인해 **매우 많은 파라미터**를 갖게 됩니다. 예를 들어, Flatten된 특징 벡터의 차원이 2048이고 이를 4096 뉴런과 연결하면 약 8백만 개 이상의 가중치가 필요합니다. 이로 인해:
- **연산량 및 메모리 사용량**이 크게 증가합니다.
- **과적합(overfitting)**의 위험이 높아지며, 이를 완화하기 위해 **Dropout**이나 **Batch Normalization** 등의 정규화 기법이 자주 사용됩니다[^5][^1].

## 5. Fully Connected Layer의 한계와 개선 방법
FC Layer는 강력한 표현력을 제공하지만, 다음과 같은 한계를 가지고 있습니다:
- **파라미터 과다**: 모든 입력과 출력이 연결되어 있어, 모델의 크기가 커지고 과적합 위험이 높아집니다.
- **공간 정보 손실**: 입력 특징의 공간적 구조를 무시하고 단일 벡터로 변환하기 때문에, 중요한 공간적 관계가 희석될 수 있습니다.

이러한 한계를 개선하기 위해 다양한 대체 기법이 제안되고 있습니다:
- **Global Average Pooling (GAP)**: 각 채널의 평균을 산출하여 FC Layer를 대체합니다. GAP는 파라미터 수를 크게 줄이고, 위치 불변성을 강화합니다.
- **Strided Convolution**: 컨볼루션 연산에 스트라이드를 적용하여 점진적으로 다운샘플링함으로써, FC Layer의 역할을 일부 대체할 수 있습니다[^4].
- **1×1 Convolution**: 채널 간 선형 결합을 통해 차원 축소 및 정보 통합을 수행하여, FC Layer의 역할을 보완합니다.
- **Transformer 기반 접근**: 최근 Vision Transformer와 같이, FC Layer 대신 어텐션 메커니즘을 활용하는 모델들이 등장하고 있습니다.

## 6. 실용적인 코드 예제

### 6.1 PyTorch를 사용한 Fully Connected Layer 구현 예제
다음은 PyTorch를 사용한 간단한 CNN 모델 예제로, 두 개의 컨볼루션 계층과 두 개의 Fully Connected Layer를 포함합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 특징 추출 계층
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # 입력: 3채널, 출력: 16채널
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # 입력: 16채널, 출력: 32채널
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 맥스 풀링
        # 분류기: Fully Connected Layer
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # 32채널 8x8 특징맵 -> 128 차원 은닉층
        self.fc2 = nn.Linear(128, 10)          # 128 차원 -> 10 클래스 출력

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
dummy_input = torch.randn(1, 3, 32, 32)  # 32x32 RGB 이미지 한 장
output = model(dummy_input)
print("PyTorch FC Model Output Shape:", output.shape)  # 예상 출력: [1, 10]
```

### 6.2 TensorFlow(Keras)를 사용한 Fully Connected Layer 구현 예제
아래는 TensorFlow(Keras)를 사용하여 유사한 CNN 모델을 구현한 예제입니다.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),  # 2D 특징맵을 1D 벡터로 변환
    layers.Dense(128, activation='relu'),  # 은닉 Fully Connected Layer
    layers.Dense(10, activation='softmax') # 출력 Fully Connected Layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

위 두 예제는 전통적인 FC Layer를 사용한 CNN 모델을 보여주며, 컨볼루션 계층에서 추출된 특징을 Flatten 후 Fully Connected Layer로 연결하여 최종 분류 결과를 출력합니다.

## 7. 최신 연구 및 트렌드
최근 연구들은 FC Layer의 **과도한 파라미터 문제**와 **공간 정보 손실**을 극복하기 위해 다양한 대체 및 최적화 방안을 제시하고 있습니다:
- **Network in Network (NiN)**: 기존 FC Layer를 1×1 Convolution과 Global Average Pooling으로 대체하여 파라미터 수를 줄이고, 모델의 일반화 성능을 향상시켰습니다[^1].
- **GoogLeNet/Inception**: 여러 컨볼루션 모듈과 Global Average Pooling을 활용하여, 전통적인 FC Layer를 대체하는 효과적인 분류기를 구현했습니다.
- **SqueezeNet**: FC Layer를 제거하고, 1×1 Convolution 및 Global Average Pooling을 통해 모델 크기를 극단적으로 경량화하였습니다[^2].
- **Vision Transformer (ViT)**: Transformer 기반 모델은 전통적인 CNN의 FC Layer를 소규모 MLP로 대체하거나 아예 제거하고, 어텐션 메커니즘으로 전역 정보를 처리하는 새로운 패러다임을 제시합니다[^3].
- **MemoryFormer**: Transformer의 Feed-Forward Network를 효율적으로 대체하는 혁신적인 방법으로, FC Layer의 연산 효율성을 극대화하는 연구가 진행되고 있습니다[^4].

이와 같이 최신 연구들은 FC Layer의 **비효율성을 줄이면서도** 모델의 표현력을 유지할 수 있는 다양한 구조적 대안을 모색하고 있으며, 특히 임베디드 및 모바일 환경에서 **모델 경량화**와 **일반화 성능 향상**에 큰 기여를 하고 있습니다.

---

# 참고문헌

[^1]: Lin, M., Chen, Q., & Yan, S. (2013). *Network in network*. **arXiv preprint arXiv:1312.4400**.
[^2]: Iandola, F., Han, S., Moskewicz, M., Ashraf, K., Dally, W., & Keutzer, K. (2016). *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size*. **arXiv preprint arXiv:1602.07360**.
[^3]: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., et al. (2021). *An image is worth 16x16 words: Transformers for image recognition at scale*. **ICLR 2021**.
[^4]: Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). *Striving for simplicity: The all convolutional net*. **arXiv preprint arXiv:1412.6806**.
[^5]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*. **Proceedings of the IEEE**, 86(11), 2278-2324.
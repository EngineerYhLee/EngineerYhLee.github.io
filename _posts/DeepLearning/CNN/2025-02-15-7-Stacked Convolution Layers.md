---
published: false
title: "(7) Stacked Convolution Layers"
description: "Stacked Convolution Layers"
header:
  teaser: /assets/images/stacked_convolution_layers.png
  og_image: /assets/images/stacked_convolution_layers.png
  image_description: "Stacked Convolution Layers"
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
  얀 르쿤과 동료들은 LeNet-5를 비롯한 초기 CNN 모델을 개발하여, 손글씨 숫자 인식(MNIST) 등 실용적 문제에서 탁월한 성능을 입증했습니다[^1].

# Stacked Convolution Layers

Stacked Convolution Layers는 여러 개의 컨볼루션(Convolution) 층을 **계층적으로 쌓아** 입력 데이터로부터 점진적으로 **저수준 특징에서 고수준 특징까지** 추출하는 방법론입니다. 단일 컨볼루션 층은 국소 영역 내 단순 패턴만 인식하는 반면, 여러 층을 쌓으면 각 층에서 비선형 활성화 함수를 통해 복잡한 패턴과 더 넓은 문맥 정보를 학습할 수 있습니다.

## 2. 개념 및 수학적 정의
Stacked Convolution Layers는 연속된 컨볼루션 연산을 통해 **계층적 특징 추출**을 가능하게 합니다. 예를 들어, \(L\)개의 컨볼루션 층을 쌓은 경우, 각 층의 필터 크기를 \(k\)라고 할 때, **유효 수용 영역(Effective Receptive Field)**은 대략 아래와 같이 계산할 수 있습니다:

$$
R = k + (k - 1)(L - 1)
$$

이는 단일 \(k \times k\) 필터가 가지는 영역보다 훨씬 넓은 범위의 정보를 한 번에 처리할 수 있음을 의미합니다. 또한, 각 컨볼루션 층 사이에 ReLU와 같은 활성화 함수가 적용되어, **비선형 변환**이 누적됨으로써 모델의 표현력이 크게 향상됩니다.

## 3. 단일 Convolution Layer와의 비교
- **단일 Convolution Layer**는 국소적인 정보만을 처리하며, 한 번의 선형 변환에 불과합니다. 따라서 복잡한 패턴 인식에는 한계가 있습니다.
- **Stacked Convolution Layers**는 여러 층을 통해 **더 깊은 계층적 표현**을 가능하게 하여, 저수준 특징(예: 엣지, 텍스처)에서부터 고수준 특징(예: 객체의 형태, 복합 패턴)까지 점진적으로 추출합니다. 이로 인해 전체 네트워크의 표현력은 극적으로 향상되지만, 그에 따른 **연산량과 파라미터 수**가 증가하는 단점도 존재합니다.

## 4. 대표적인 Stacked Convolution 구조
- **VGGNet (2014)**: VGG는 **3×3 Conv Layers**를 반복적으로 쌓아 깊은 네트워크를 구성하는 방식을 도입했습니다. 여러 개의 작은 필터를 연속 사용하여 큰 필터와 동일한 수용 영역을 얻으면서도, 파라미터 수를 크게 줄이는 장점이 있습니다[^3].
  
- **ResNet (2016)**: ResNet은 **Residual Connection (Skip Connection)**을 도입하여, 매우 깊은 네트워크에서도 효과적인 학습을 가능하게 했습니다. 잔차 블록 내에 여러 컨볼루션 층을 쌓고 입력을 출력에 더해줌으로써 기울기 소실 문제를 완화하였습니다[^4].
  
- **DenseNet (2017)**: DenseNet은 모든 이전 층의 출력을 현재 층에 **Concatenation**하여 전달함으로써, 각 층이 풍부한 정보를 재활용하도록 설계되었습니다. 이 구조는 깊은 네트워크에서도 학습이 원활하도록 돕고 파라미터 효율을 극대화합니다[^5].
  
- **MobileNet & EfficientNet**: 경량 모델에서는 Depthwise Separable Convolution과 Compound Scaling 기법을 활용하여, Stacked Convolution Layers의 깊이를 유지하면서도 연산량과 파라미터 수를 획기적으로 줄이는 구조가 연구되고 있습니다[^6].

## 5. 학습 효과 및 연산량 분석
Stacked Convolution Layers를 깊게 쌓을수록 모델의 **표현력**은 증가하지만, 동시에 다음과 같은 트레이드오프가 발생합니다:
- **파라미터 증가**: 단순히 층을 추가하면 각 층의 가중치가 누적되어 모델 용량이 증가합니다. 예를 들어, VGG-16은 13개의 컨볼루션 층과 3개의 Fully Connected Layer로 구성되어 수억 개의 파라미터를 가지지만, Residual Block을 사용하는 ResNet은 깊이 대비 파라미터 효율이 뛰어납니다.
- **연산량(FLOPs) 증가**: 깊은 네트워크는 순전파 및 역전파 시 많은 연산을 필요로 하므로, 학습 및 추론 시간이 늘어납니다.
- **학습 안정성**: 층이 너무 깊으면 기울기 소실 또는 폭발 문제가 발생할 수 있으나, Residual Connection이나 Dense Connectivity 등의 기법을 통해 이를 완화할 수 있습니다.
- **일반화 성능**: 적절히 깊은 모델은 더 복잡한 특징을 학습할 수 있으나, 과도한 깊이는 과적합의 위험을 내포하므로, 정규화 기법(Dropout, Batch Normalization)과 충분한 학습 데이터가 필요합니다.

## 6. 최신 연구 및 트렌드
최근 연구들은 Stacked Convolution Layers의 효과를 극대화하고 효율성을 개선하기 위해 다양한 기법을 도입하고 있습니다:
- **ConvNeXt (CVPR 2022)**: 기존 ResNet 구조를 현대적으로 리팩토링하여, 단순한 컨볼루션 기반 모델이 Transformer에 필적하는 성능을 낼 수 있음을 보여주었습니다[^7].
- **RepLKNet & SLaK (ICLR 2023)**: 매우 큰 커널을 희소화 기법과 결합하여 사용함으로써, 전통적인 Stacked Convolution 구조와는 다른 방식으로 넓은 receptive field를 직접 확보하는 연구들이 진행되고 있습니다[^8].
- **Neural Architecture Search (NAS)**: AutoML 기법을 통해 최적의 컨볼루션 층 수와 구조를 자동으로 결정하는 연구들이 활발히 이루어지고 있으며, 이를 통해 효율과 성능의 트레이드오프를 최적화하고 있습니다[^6].
- **컨볼루션과 Attention의 결합**: Conv와 Self-Attention 모듈을 결합한 하이브리드 모델들이 제안되어, 지역적 특징 추출과 전역적 정보 통합을 동시에 수행하는 모델이 등장하고 있습니다.

## 7. 실용적인 코드 예제

### 7.1 PyTorch 예제
아래 예제는 PyTorch를 사용하여 **Stacked Convolution Layers**로 구성된 간단한 CNN 모델을 구현한 것입니다. `StackedCNN` 모델은 두 개 이상의 연속된 컨볼루션 층을 사용하여 특징을 추출한 후, 풀링과 Fully Connected Layer로 분류를 수행합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StackedCNN, self).__init__()
        # Stacked Convolution Layers: 3개의 연속 컨볼루션 층
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Conv1
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # Conv2
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # Conv3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # 다운샘플링
        )
        # Fully Connected Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 모델 생성 및 테스트
model = StackedCNN(num_classes=10)
dummy_input = torch.randn(1, 3, 32, 32)
output = model(dummy_input)
print("PyTorch StackedCNN Output Shape:", output.shape)
```

### 7.2 TensorFlow(Keras) 예제
다음은 TensorFlow(Keras)를 사용하여 동일한 개념의 Stacked Convolution Layers 모델을 구현한 예제입니다.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_stacked_cnn(input_shape=(32,32,3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

stacked_cnn = create_stacked_cnn()
stacked_cnn.summary()
```

위 예제에서, Keras 모델은 3개의 연속된 Conv2D 층을 사용한 후 MaxPooling, Flatten, 그리고 두 개의 Dense Layer로 구성되어 있습니다. `model.summary()`를 통해 모델 구조와 파라미터 수를 확인할 수 있으며, 이를 통해 Stacked Convolution Layers가 단일 컨볼루션층보다 **더 복잡한 특징 추출**과 **향상된 분류 성능**을 가능하게 함을 알 수 있습니다.

---

# 참고문헌

[^1]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*. **Proceedings of the IEEE**, 86(11), 2278-2324.
[^2]: Hubel, D. H., & Wiesel, T. N. (1959). *Receptive fields of single neurones in the cat's striate cortex*. **The Journal of Physiology**, 148(3), 574-591.
[^3]: Simonyan, K., & Zisserman, A. (2014). *Very deep convolutional networks for large-scale image recognition*. **arXiv preprint arXiv:1409.1556**.
[^4]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. **CVPR**, 770-778.
[^5]: Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely connected convolutional networks*. **CVPR**, 4700-4708.
[^6]: Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks*. **ICML**, 6105-6114.
[^7]: Liu, Z., Mao, H., Wu, Y., et al. (2022). *A ConvNet for the 2020s*. **CVPR 2022**.
[^8]: RepLKNet authors. (2022). *Large Kernel Matters -- Improve Semantic Segmentation by Globalizing Local Convolutions*. **arXiv preprint arXiv:2203.06717**.
---
published: false
title: "[CNN] (7) 컨볼루션 레이어 쌓기(Stacked Convolutional Layers)"
description: "Stacked Convolutional Layers"
header:
  teaser: /assets/images/stacked_convolution_layers.png
  og_image: /assets/images/stacked_convolution_layers.png
  image_description: "Stacked Convolutional Layers"
date: 2025-02-18
last_modified_at: 2025-02-18 18:00:00
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

이번 포스팅에서는 컨볼루션 레이어 쌓기에 대해 자세히 알아보겠습니다.

## CNN 시리즈
1. [Convolutional Neural Networks 개요]({{ site.url }}{{ site.baseurl }}/cnn/1_CNN-Basics/)
2. [CNN의 역사와 발전 과정, 주요 모델들]({{ site.url }}{{ site.baseurl }}/cnn/2_CNN-History/)
3. [컨볼루션 레이어(Convolutional Layer)]({{ site.url }}{{ site.baseurl }}/cnn/3_Convolution-Layer/)
4. [패딩(Padding)과 스트라이드(Stride)]({{ site.url }}{{ site.baseurl }}/cnn/4_Padding-and-Stride/)
5. [풀링 레이어(Pooling Layer)]({{ site.url }}{{ site.baseurl }}/cnn/5_Pooling-Layers/)
6. [정규화 레이어(Normalization Layer)]({{ site.url }}{{ site.baseurl }}/cnn/6_Normalization-Layer/)
<!-- 7. [컨볼루션 레이어 쌓기(Stacked Convolutional Layers)]({{ site.url }}{{ site.baseurl }}/cnn/7_Stacked-Convolution-Layers/) -->


[//]: # (작성 양식)
[//]: # (# 마크다운: kramdown)
[//]: # (# 수식 예시 => 수식이 필요할 경우에만 작성:  $$y = f(z) = f\left(\sum_{i=1}^n w_i x_i + b\right)$$)
[//]: # (# 언어: 한글, 주요 단어는 영문과 같이 작성)
[//]: # (# 참고문헌: 실제 존재하는 논문들을 형식에 맞게 기재)

## 1. 인용 예시 
- **LeNet-5 등장**  
  얀 르쿤과 동료들은 LeNet-5를 비롯한 초기 CNN 모델을 개발하여, 손글씨 숫자 인식(MNIST) 등 실용적 문제에서 탁월한 성능을 입증했습니다[^1].

# Stacked Convolutional Layers

Stacked Convolutional Layers는 여러 개의 컨볼루션(Convolution) 층을 **순차적으로 쌓아** 입력 데이터로부터 **계층적 특징 추출(Hierarchical Feature Extraction)**을 수행하는 방법론입니다. 단일 Convolutional Layer가 로우-레벨(Low-level) 특징(엣지, 코너 등)에 집중한다면, 층을 거듭 쌓으면서 점진적으로 더 복잡한 하이-레벨(High-level) 특징(객체의 부분, 형태 등)을 학습할 수 있습니다. 또한, 이러한 다중 컨볼루션 층을 통해 **효과적 수용 영역(Effective Receptive Field)**이 확대되어, 넓은 맥락 정보를 함께 고려할 수 있게 됩니다. 이는 현대 CNN 구조에서 높은 성능을 달성하는 핵심 원리 중 하나입니다.

## 2. 개념 및 수학적 정의
Stacked Convolutional Layers의 핵심은 **컨볼루션 연산**이 여러 차례 연쇄적으로 적용된다는 점에 있습니다. 간단히 표현하면, \(L\)개의 컨볼루션 연산과 활성화 함수가 겹겹이 쌓여 다음과 같은 형태가 됩니다:

\[
h_1(x) = f_1(\mathrm{Conv}_1(x)), \quad
h_2(x) = f_2(\mathrm{Conv}_2(h_1(x))), \quad \dots \quad
h_L(x) = f_L(\mathrm{Conv}_L(h_{L-1}(x))).
\]

이때 \(f_i\)는 ReLU 등 비선형 활성화 함수를 의미합니다. 여러 층을 거듭 적용하면 수용 영역이 누적되어 더 넓은 영역에서 정보를 통합할 수 있습니다. 모든 컨볼루션의 커널 크기가 \(k\)라고 하고, 스트라이드가 1이라고 가정한다면, \(L\)번의 연속된 컨볼루션을 거친 후 **유효 수용 영역** \(R\)은 대략 아래와 같이 계산됩니다:

$$
R = k + (k - 1)(L - 1).
$$

예를 들어, \(k=3\) ( \(3\times3\) )인 필터를 3개 스택으로 쌓으면 \(R\)은 \(7\times7\) 영역에 해당하므로, 단일 큰 커널(\(7\times7\))을 한 번 적용한 것과 유사한 수용 영역을 얻게 됩니다. 하지만 이 과정에서 **층마다 활성화 함수가 중첩**되므로, 단일 큰 커널을 한 번 쓰는 것보다 훨씬 **비선형 표현력**이 커지는 장점을 얻을 수 있습니다.

## 3. 단일 Convolutional Layer와의 비교
- **단일 Convolutional Layer**:  
  하나의 필터(\(k\times k\))로 한 번의 컨볼루션과 단일 활성화만 적용하므로, 로컬한 특징만 추출하고 모델 복잡도가 상대적으로 낮습니다. 또한 수용 영역이 한 번에 확장되는 크기에 한계가 있으며, 학습할 수 있는 표현의 범위가 제한적입니다.

- **Stacked Convolutional Layers**:  
  여러 층을 순차적으로 적용하므로, **계층적 특징 추출**과 함께 **광범위한 수용 영역**을 확보할 수 있습니다. 특히 작은 커널(예: \(3\times3\))을 여러 번 쌓을 경우, 큰 커널과 동일한 수용 영역을 얻으면서도 파라미터 수가 줄고 비선형성이 누적되어 **더 높은 표현력**을 갖게 됩니다. 단, 층이 많아질수록 기울기 소실(Vanishing Gradient) 문제나 연산량(연산 복잡도) 증가에 유의해야 합니다.

## 4. 대표적인 Stacked Convolution 구조
- **VGGNet (2014)**  
  VGG는 모두 \(3\times3\) 크기의 Conv Layer를 여러 번 연속해서 적용하는 방식을 채택했습니다. 예를 들어, \(5\times5\) 컨볼루션을 한 번 사용하는 대신 \(3\times3\) 컨볼루션 두 번을 스택으로 쌓아 동일한 수용 영역을 달성하되, 파라미터는 줄이고 비선형 활성화(예: ReLU) 사용을 늘리는 장점을 얻었습니다[^3].  

- **ResNet (2016)**  
  ResNet은 **Residual Connection (Skip Connection)**을 통해 매우 깊은 네트워크를 안정적으로 학습할 수 있도록 한 대표적 모델입니다. 컨볼루션 블록 사이에 입력을 직접 출력으로 더해주는 경로를 추가함으로써, 기울기 소실 문제를 완화하고 100층 이상의 극딥 네트워크까지도 학습이 가능해졌습니다[^4].  

- **DenseNet (2017)**  
  DenseNet은 모든 이전 층의 출력을 현재 층에 **Concatenation**하여 전달하는 **밀집 연결(Dense Connectivity)** 아이디어를 도입했습니다. 이렇게 하면 각 층이 기존 특징들을 재활용하여 학습 효율이 높아지고, 기울기가 원활하게 전파되어 학습 안정성도 향상됩니다[^5].  

- **MobileNet & EfficientNet**  
  제한된 자원 환경에서 모델을 경량화하기 위해, **Depthwise Separable Convolution**(MobileNet)이나 **Compound Scaling**(EfficientNet) 기법 등을 활용하여 연산량과 파라미터 수를 크게 줄이면서도 Stacked Convolutional Layers의 효과를 최대한 유지하는 방법이 연구되고 있습니다[^6].  

## 5. 학습 효과 및 연산량 분석
Stacked Convolutional Layers를 깊게 쌓을수록 **더 풍부한 계층적 표현**을 학습할 수 있으나, 동시에 다음과 같은 트레이드오프가 존재합니다:

- **기울기 소실/폭발 (Vanishing/Exploding Gradients)**  
  층이 깊어질수록 역전파되는 기울기가 매우 작아지거나(소실), 너무 커질 위험(폭발)이 있습니다. Residual Connection, Batch Normalization 등의 기법을 통해 이를 완화할 수 있습니다.

- **연산량(FLOPs) 증가 및 메모리 사용량 증가**  
  스택된 층이 많아질수록 컨볼루션 연산과 중간 특징맵 저장에 필요한 메모리가 기하급수적으로 늘어납니다. 따라서 실제 응용에서는 모델 경량화나 하드웨어 가속 등이 매우 중요합니다.

- **과적합(Overfitting) 위험**  
  깊은 모델은 파라미터가 매우 많아, 훈련 데이터에 과도하게 적합하기 쉬우므로, 정규화 기법(Dropout, Weight Decay 등)과 데이터 증강(Data Augmentation)이 필수적입니다.

- **파라미터 효율성과 표현력 간 균형**  
  단순히 깊이만 늘리기보다는, 채널 너비나 입력 해상도, 커널 크기 등을 조합하는 **스케일링 전략**(예: EfficientNet)이나 **하이브리드 구조**(예: 컨볼루션+Attention)로 설계하면, 더 효율적이면서도 높은 성능을 얻을 수 있습니다.

## 6. 최신 연구 및 트렌드
- **ConvNeXt (CVPR 2022)**  
  ResNet 계열의 전통적 ConvNet을 현대적으로 재설계하여, Vision Transformer에 버금가는 성능을 달성했습니다. 예컨대 더 큰 커널(7×7), Layer Normalization 사용, GELU 활성화 함수 적용 등으로 단순함과 효율성을 모두 만족합니다[^7].

- **RepLKNet & SLaK (ICLR 2023)**  
  **초대형 커널**(예: 31×31)을 사용하는 컨볼루션 구조를 re-parameterization 기법과 결합해, 넓은 수용 영역을 직접 확보하는 전략이 제안되었습니다. 이는 작은 커널을 많이 쌓는 전통적 방식과 다른 방식으로 **전역적 형태 정보**를 효과적으로 학습합니다[^8].

- **Neural Architecture Search (NAS)**  
  AutoML 기법으로 CNN 구조(층 수, 커널 크기, 채널 너비 등)를 자동 탐색하여, 정확도와 효율 간 최적 절충을 찾는 시도가 활발합니다. EfficientNet 역시 NAS로 찾은 기본 블록에 Compound Scaling을 적용해 높은 효율을 달성했습니다[^6].

- **컨볼루션과 Attention의 결합**  
  Conv와 Self-Attention을 함께 쌓는 **하이브리드 모델**(예: CoAtNet)도 연구되고 있습니다. 이는 국소 패턴 학습에 강한 Convolution과 전역 의존성을 포착하는 Attention의 장점을 모두 취하려는 시도로, 대규모 데이터부터 소규모 데이터까지 폭넓게 적용되고 있습니다.

## 7. 실용적인 코드 예제

### 7.1 PyTorch 예제
아래 예제는 PyTorch를 사용하여 **Stacked Convolutional Layers**를 세 개 연속으로 적용한 간단한 CNN을 구현한 것입니다. 첫 번째 컨볼루션 층의 출력을 두 번째 컨볼루션 층이 입력으로 받고, 다시 세 번째 층까지 연결되어 계층적으로 특징을 추출합니다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StackedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StackedCNN, self).__init__()
        # Stacked Convolutional Layers: 3개의 연속 컨볼루션 층
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
다음 예제는 TensorFlow(Keras)를 사용하여 유사한 구성의 **Stacked Convolutional Layers**를 구현한 코드입니다.

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

여기서도 3개의 Conv2D 층을 연속으로 쌓은 뒤, MaxPooling, Flatten, Dense Layer를 거쳐 최종 분류를 수행합니다. 이처럼 **Stacked Convolutional Layers**를 통해 계층적으로 다양한 수준의 특징을 추출할 수 있으며, 모델 구조(층의 개수, 채널 수 등)를 조정해가며 원하는 복잡도와 정확도 사이의 균형을 찾을 수 있습니다.

---

# 참고문헌

[^1]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-based learning applied to document recognition*. **Proceedings of the IEEE**, 86(11), 2278-2324.  
[^2]: Hubel, D. H., & Wiesel, T. N. (1959). *Receptive fields of single neurones in the cat's striate cortex*. **The Journal of Physiology**, 148(3), 574-591.  
[^3]: Simonyan, K., & Zisserman, A. (2014). *Very deep convolutional networks for large-scale image recognition*. **arXiv preprint arXiv:1409.1556**.  
[^4]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. **CVPR**, 770-778.  
[^5]: Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). *Densely connected convolutional networks*. **CVPR**, 4700-4708.  
[^6]: Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks*. **ICML**, 6105-6114.  
[^7]: Liu, Z., Mao, H., Wu, Y., et al. (2022). *A ConvNet for the 2020s*. **CVPR**.  
[^8]: Ding, X., Zhang, X., Ma, N., et al. (2022). *Scaling up your kernels to 31x31: Revisiting large kernel design in CNNs*. **arXiv preprint arXiv:2203.06717**.  
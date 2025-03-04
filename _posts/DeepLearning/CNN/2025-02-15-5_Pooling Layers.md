---
published: true
title: "[CNN] (5) 풀링 레이어(Pooling Layer)"
description: "CNN에서 풀링 레이어의 역할과 종류, 그리고 최신 동향을 살펴봅니다."
header:
  teaser: /assets/images/max_avg_pooling.png
  og_image: /assets/images/max_avg_pooling.png
  image_description: "Pooling Layers"
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
  - Pooling
---

이번 포스팅에서는 CNN의 중요한 구성 요소인 **풀링 레이어(Pooling Layer)**에 대해 자세히 알아보겠습니다.  

## CNN 시리즈
1. [Convolutional Neural Networks 개요]({{ site.url }}{{ site.baseurl }}/cnn/1_CNN-Basics/)
2. [CNN의 역사와 발전 과정, 주요 모델들]({{ site.url }}{{ site.baseurl }}/cnn/2_CNN-History/)
3. [컨볼루션 레이어(Convolutional Layer)]({{ site.url }}{{ site.baseurl }}/cnn/3_Convolution-Layer/)
4. [패딩(Padding)과 스트라이드(Stride)]({{ site.url }}{{ site.baseurl }}/cnn/4_Padding-and-Stride/)
5. [풀링 레이어(Pooling Layer)]({{ site.url }}{{ site.baseurl }}/cnn/5_Pooling-Layers/)
6. [정규화 레이어(Normalization Layer)]({{ site.url }}{{ site.baseurl }}/cnn/6_Normalization-Layer/)
<!-- 7. [컨볼루션 레이어 쌓기(Stacked Convolutional Layers)]({{ site.url }}{{ site.baseurl }}/cnn/7_Stacked-Convolution-Layers/) -->

---

## 1. 풀링의 기본 개념

### 1.1. 풀링이란?
- 풀링(Pooling)은 컨볼루션 신경망(CNN)에서 **Feature Map**의 공간적 크기를 줄이는 **다운샘플링(Downsampling)** 연산을 의미합니다.  
- 컨볼루션(Convolution)은 입력 Feature의 **위치 정보**를 보존합니다. 이를 **등변성(Equivariance)**이라 합니다.  
- 위치 변화에도 **불변성(Invariance)**을 얻기 위해 **Pooling**을 적용할 수 있습니다. Pooling을 적용하면 입력의 세부 위치가 조금씩 달라져도 비슷한 Feature를 얻을 수 있습니다.

![image_classification](/assets/images/CNN/image_classification.png)

*A simple CNN for classifying images. [^6]*

### 1.2. 풀링의 목적
1. **계산 효율성 향상**  
   - 풀링으로 Feature Map의 해상도를 낮춰 **연산량**과 **메모리 사용량**을 절감합니다.
2. **과적합(Overfitting) 방지**  
   - 풀링 영역 내 특징들을 요약하여 불필요한 세부 정보를 제거함으로써 모델의 **일반화(Generalization)**를 돕습니다.
3. **위치 불변성(Invariance) 제공**  
   - 입력 이미지가 소폭 이동하거나 왜곡되더라도, 주요 특징을 유지해 **강인성(Robustness)**을 부여합니다.

### 1.3. 풀링의 연산 방식
- 풀링 연산자(Pooling Operator)는 입력의 모든 영역을 스트라이드에 따라 슬라이드하여 풀링 윈도우(Pooling Window)를 통과하는 각 위치에 대해 단일 출력을 계산합니다.
- 컨볼루션 레이어의 입력과 커널 간의 교차 상관(Cross-Correlation) 계산과 달리 풀링 레이어에는 커널과 매개변수가 없습니다. 
- 대신 풀링 연산자는 일반적으로 풀링 윈도우 내의 요소들 중 최대값 또는 평균값을 계산합니다.(Max Pooling, Average Pooling)

### 1.4. 풀링의 수학적 표현
- 일반적으로 $$ h \times w $$ 크기의 풀링 윈도우(Pooling Window)를 사용한다고 할 때, 입력의 높이를 $$H_{in}$$, 너비를 $$W_{in}$$라 하면, 출력의 크기 $$ (H_{out}, W_{out}) $$는 다음과 같이 계산됩니다.  

  $$
  H_{out} = \left\lceil \frac{H_{in}}{h} \right\rceil, 
  \quad
  W_{out} = \left\lceil \frac{W_{in}}{w} \right\rceil
  $$


---

## 2. 풀링의 종류

### 2.1. Max Pooling
#### 2.1.1. Max Pooling의 정의
- 풀링 윈도우 내에서 **최대값**을 추출하는 연산
- 각 윈도우 내에서 가장 큰 값을 선택하므로 이미지 내에서 가장 강하게 활성화된 특징(예: 경계, 모서리 등)을 효과적으로 추출
- **세부 정보 손실**과 **위치 정보 소실**이 발생할 수 있음

#### 2.1.2. Max Pooling의 수식

$$
y_{ij} = \max_{(a,b) \in R_{ij}} x_{ab}
$$

- **$$y_{ij}$$**: 풀링 연산의 결과로 생성되는 **출력(Feature Map)의 한 위치** $$(i, j)$$에 해당하는 값
- **$$\max_{(a,b) \in R_{ij}} x_{ab}$$**: $$R_{ij}$$ 영역 내의 모든 값들 $$x_{ab}$$ 중에서 **최대값**을 선택
- **$$R_{ij}$$**: 입력 데이터 $$x$$의 특정 부분으로 **풀링 윈도우**를 의미
- **$$x_{ab}$$**: 입력 데이터 $$x$$의 위치로 **$$(a, b)$$에 있는 값**

![max_pooling](/assets/images/CNN/max_pooling.png)

*Illustration of Max Pooling (2x2) with a stride of 1. [^5]*

### 2.2. Average Pooling

#### 2.2.1. Average Pooling의 정의
- 풀링 윈도우 내 모든 값을 **평균**으로 추출하는 연산
- 풀링 영역 내 모든 값을 고려하여 평균을 내므로 각 영역의 전체적인 특성을 반영
- 전체 영역의 평균값을 사용하므로 일부 노이즈가 완화되는 효과
- 강하게 활성화된 중요한 특징이 평균에 의해 희석될 수 있음

#### 2.2.2. Average Pooling의 수식

$$
y_{ij} = \frac{1}{|R_{ij}|} \sum_{(a,b) \in R_{ij}} x_{ab}
$$

- **$$y_{ij}$$**: 출력 Feature Map의 $$(i, j)$$ 위치에 해당하는 값
- **$$R_{ij}$$**: 입력 Feature Map에서 선택된 풀링 윈도우 영역을 의미
- **$$\mid R_{ij} \mid$$**: 풀링 윈도우 $$R_{ij}$$에 포함된 요소의 총 개수
- **$$\sum_{(a,b) \in R_{ij}} x_{ab}$$**: 풀링 윈도우 내의 모든 값 $$x_{ab}$$의 합을 의미


![max_avg_pooling](/assets/images/CNN/max_avg_pooling.png)

*Illustration of Max Pooling and Average Pooling Figure 2 above shows an example of max pooling operation and average pooling with a 2x2 pixel filter size from 4x4 pixel input. At max pooling, each filter is taken the maximum value, then arranged into a new output with a size of 2x2 pixels. While the average pooling value taken is the average value of the filter size. Classification layer is a layer consisting of flattening, hidden layer and activation functions. Hidden layers in artificial neural networks is layers between input layer and output layer, where artificial neurons take a set of weight inputs and produce output through activation functions such as sigmoid[8], ReLU[9], or Softmax[10]. [^7]*


---

## 3. 풀링의 한계와 대체 기법

### 3.1. 풀링의 한계
-  **정보 손실**:하나의 대표값(최댓값 혹은 평균)만 남기므로, 세부 정보가 상당 부분 사라집니다.
-  **고정된 요약 방식**: 맥스 풀링은 극단값, 평균 풀링은 고른 반영 방식으로 **데이터 특성에 맞는 적응형 요약**에 제약이 있습니다.

### 3.2. 대체 기법

- **글로벌 풀링(Global Pooling)**  
   - Feature Map 전체 영역에 대해 맥스나 평균 연산을 수행  
   - 예: **Global Average Pooling**은 다양한 입력 크기를 같은 크기의 출력 벡터로 변환하여 **파라미터 수 감소**와 **유연성**을 동시에 확보

- **스트라이드 컨볼루션(Strided Convolution)**  
   - 풀링을 사용하지 않고, 컨볼루션에서 **Stride**를 크게 설정해 **다운샘플링** 효과를 얻는 방식  
   - 학습 가능한 필터를 사용하므로 단순 풀링보다 **세밀한 정보**를 더 보존할 수 있음 [^4]

- **어텐션 기반 풀링(Attention-based Pooling)**  
   - 간단한 최댓값/평균이 아닌, **학습된 어텐션 가중치**를 통해 중요한 픽셀을 강조하고 덜 중요한 정보는 축소  
   - **적응형**으로 특징을 요약하므로 다양한 상황에서 효과적

- **하이브리드/적응형 풀링(Adaptive Pooling) & Fractional Pooling**  
   - 맥스/평균을 혼합하거나, 풀링 윈도우를 유연하게 변경하는 **Fractional Max Pooling**도 제안됨 [^1]  
   - 예: PyTorch의 `AdaptiveAvgPool2d`로 임의 입력을 고정된 크기로 변환 가능  
   - Fractional Max Pooling은 일반 정수 크기 풀링 대신 비정수 크기를 사용해 **개별 단계에서의 다운샘플링 비율**을 미세 조정

---


## 4. 예시 코드

아래 파이썬 코드는 **Max Pooling**과 **Average Pooling**을 적용해서 2D 컨볼루션 연산을 진행한 예시입니다.

```python
import torch
import torch.nn.functional as F
import pandas as pd

if __name__ == "__main__":
    # 4x4 입력 이미지 (예시)
    input_image = torch.tensor(
        [
            [0, 0, 1, 5],
            [0, 1, 5, 1],
            [1, 5, 1, 0],
            [5, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    image_tensor = input_image.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    max_pool_output = F.max_pool2d(image_tensor, kernel_size=2).squeeze()
    avg_pool_output = F.avg_pool2d(image_tensor, kernel_size=2).squeeze()

    print("입력 이미지:")
    print(pd.DataFrame(input_image.int().tolist()).to_string(index=False, header=False))

    print("\nMax Pooling (pool_size=2):")
    print(
        pd.DataFrame(max_pool_output.int().tolist()).to_string(
            index=False, header=False
        )
    )

    print("\nAverage Pooling (pool_size=2):")
    print(
        pd.DataFrame(avg_pool_output.int().tolist()).to_string(
            index=False, header=False
        )
    )

```

```console
# 출력 결과
입력 이미지:
0 0 1 5
0 1 5 1
1 5 1 0
5 1 0 0

Max Pooling (pool_size=2):
1 5
5 1

Average Pooling (pool_size=2):
0 3
3 0
```

위 코드 결과를 보면 Max Pooling과 Average Pooling의 차이를 확인할 수 있습니다.





---

## 참고문헌

[^1]: Graham, B. (2014). **Fractional max-pooling**. arXiv preprint arXiv:1412.6071.  
[^2]: Lin, M., Chen, Q., & Yan, S. (2013). **Network in network**. arXiv preprint arXiv:1312.4400.  
[^4]: Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). **Striving for simplicity: The all convolutional net**. arXiv preprint arXiv:1412.6806.  
[^5]: Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2021). **Dive into deep learning**. arXiv preprint arXiv:2106.11342.  
[^6]: Murphy, K. P. (2012). **Machine learning: a probabilistic perspective**. MIT press.
[^7]: Yani, M., Budhi Irawan, S. S. M., & Casi Setiningsih, S. M. (2019). Application of transfer learning using convolutional neural network method for early detection of terry’s nail. In Journal of Physics: Conference Series (Vol. 1201, No. 1, p. 012052). IOP Publishing.
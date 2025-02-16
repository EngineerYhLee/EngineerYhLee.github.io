---
published: false
title: "[CNN] (5) 풀링 계층(Pooling Layer)"
description: "CNN에서 풀링 계층의 역할과 종류, 그리고 최신 동향을 살펴봅니다."
header:
  teaser: /assets/images/pooling_layers.png
  og_image: /assets/images/pooling_layers.png
  image_description: "Pooling Layers"
date: 2025-02-15
last_modified_at: 2025-02-15 22:00:00
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

이번 포스팅에서는 CNN의 중요한 구성 요소인 **풀링 계층(Pooling Layer)**에 대해 자세히 알아보겠습니다.

## CNN 시리즈
1. [Convolutional Neural Networks 개요]({{ site.url }}{{ site.baseurl }}/cnn/1_CNN-Basics/)
2. [CNN의 역사와 발전 과정, 주요 모델들]({{ site.url }}{{ site.baseurl }}/cnn/2_CNN-History/)
3. [컨볼루션 레이어(Convolutional Layer)]({{ site.url }}{{ site.baseurl }}/cnn/3_Convolution-Layer/)
<!-- 4. [패딩(Padding)과 스트라이드(Stride)]({{ site.url }}{{ site.baseurl }}/cnn/4_Padding-and-Stride/) -->
<!-- 5. [풀링 레이어(Pooling Layer)]({{ site.url }}{{ site.baseurl }}/cnn/5_Pooling-Layers/) -->
<!-- 6. [다중 채널(Multiple Channels)]({{ site.url }}{{ site.baseurl }}/cnn/6_Multiple-Channels/) -->
<!-- 7. [컨볼루션 레이어 쌓기(Stacked Convolutional Layers)]({{ site.url }}{{ site.baseurl }}/cnn/7_Stacked-Convolution-Layers/) -->

---

# 풀링의 기본 개념

## 1. 풀링의 정의와 목적
풀링(Pooling)은 컨볼루션 신경망(CNN)에서 **Feature Map**의 공간적 크기를 줄이는 **다운샘플링(Downsampling)** 연산을 의미합니다. 주요 목적은 다음과 같습니다:

1. **계산 효율성 향상**  
   - 풀링으로 Feature Map의 해상도를 낮춰 **연산량**과 **메모리 사용량**을 줄입니다.
2. **과적합 방지**  
   - 풀링 영역 내 특징들을 요약하여 불필요한 세부 정보를 제거함으로써, 모델의 **일반화(Generalization)** 능력을 높입니다.
3. **위치 불변성(Invariance) 제공**  
   - 입력 이미지가 소폭 이동하거나 왜곡되더라도, 중요한 특징은 유지되도록 하여 **강인성(Robustness)**을 부여합니다.

---

## 2. 풀링의 수학적 표현
일반적으로 \( h \times w \) 크기의 풀링 윈도우(Window)를 사용할 때, 입력의 높이를 \(H_{in}\), 너비를 \(W_{in}\)라 하면, 출력의 크기 \( (H_{out}, W_{out}) \)는 다음과 같이 계산됩니다:

\[
H_{out} = \left\lceil \frac{H_{in}}{h} \right\rceil, 
\quad
W_{out} = \left\lceil \frac{W_{in}}{w} \right\rceil
\]

여기서 \(\lceil \cdot \rceil\)는 올림(ceiling) 연산을 의미합니다.  
또한, 풀링이 일반화된 형태인 **Lp 풀링**은 다음 식으로 표현됩니다:

\[
s_{ij} = \left(\frac{1}{|R_{ij}|} \sum_{(a,b) \in R_{ij}} x_{ab}^p \right)^{1/p}
\]

- \(R_{ij}\): 풀링 영역(윈도우)  
- \(p=1\)이면 **평균 풀링(Average Pooling)**과 유사  
- \(p \to \infty\)이면 **맥스 풀링(Max Pooling)**과 유사한 효과

이처럼 \(p\) 값을 조절하여 **평균 풀링**과 **맥스 풀링** 사이의 효과를 다양하게 구현할 수 있습니다 [^1][^5].

---

# 풀링의 종류

## 1. 최대 풀링 (Max Pooling)
- **정의**: 풀링 윈도우 내에서 **최대값**만 추출하는 방식  
- **장점**: 강하게 활성화된 (뚜렷한) 특징만 남기므로, **선명한 Feature**를 얻을 수 있음  
- **단점**: 윈도우 내 다른 값들은 모두 무시되므로, **일부 정보 손실**이 발생

\[
y_{ij} = \max_{(a,b) \in R_{ij}} x_{ab}
\]

---

## 2. 평균 풀링 (Average Pooling)
- **정의**: 풀링 윈도우 내 모든 값을 **평균**으로 요약  
- **장점**: **모든** 정보를 고르게 반영해, **부드러운 Feature Map**을 얻을 수 있음  
- **단점**: 강한 특징이 희석되어, **뚜렷한 특징 추출 능력**이 떨어질 수 있음

\[
y_{ij} = \frac{1}{|R_{ij}|} \sum_{(a,b) \in R_{ij}} x_{ab}
\]

---

## 3. Lp 풀링 (Generalized Pooling)
- **정의**: \( p \)-놈( \(\|\cdot\|_{p}\) )을 이용해 풀링 연산을 확장한 방식  
- **특징**:  
  - \( p=1 \)일 때 평균 풀링과 동일  
  - \( p \rightarrow \infty \)일 때 맥스 풀링과 유사  
  - **맥스와 평균 사이를 조절**할 수 있어, 상황에 따른 유연성 제공

\[
s_{ij} = \left(\frac{1}{|R_{ij}|} \sum_{(a,b) \in R_{ij}} x_{ab}^p \right)^{1/p}
\]

---

# 풀링이 신경망 학습 및 연산량 감소에 미치는 영향
1. **수용 영역(Receptive Field) 확장**  
   - 풀링을 적용하면 출력의 해상도가 내려가므로, 후속 레이어에서 한 픽셀이 **더 넓은 범위**의 정보를 포괄하게 됩니다.  
2. **연산량 감소**  
   - 풀링으로 Feature Map을 축소하면, 이후 컨볼루션이나 완전연결층(FC Layer) 연산에 필요한 **입력 차원**이 줄어듭니다.  
3. **과적합 방지 및 일반화 강화**  
   - 풀링은 세부 정보를 단순화하여 **불필요한 잡음(Noise)을 제거**하고, 모델이 입력의 사소한 변형에 덜 민감하도록 합니다 [^5].

---

# 풀링의 한계와 대체 기법

## 1. 풀링의 한계
- **정보 손실**  
  - 풀링 윈도우 내 전체 값을 하나의 요약 값(최댓값 혹은 평균)으로 대체하므로, 원래의 세부 정보가 사라집니다.
- **고정된 요약 방식**  
  - 맥스 풀링은 극단값만 취하고, 평균 풀링은 모든 값을 균등 가중으로 취하므로, **데이터 특성**에 맞는 **적응형 요약**이 어렵습니다.

---

## 2. 대체 기법

1. **글로벌 풀링(Global Pooling)**  
   - Feature Map 전체 영역에 대해 맥스나 평균 연산을 수행해, **하나의 스칼라**로 요약하는 기법  
   - 예: **Global Average Pooling**을 활용하면, 다양한 입력 크기에 대해 동일한 크기의 출력 벡터를 얻을 수 있어 **파라미터 수 감소**와 **유연성**을 동시에 확보합니다.

2. **스트라이드 컨볼루션(Strided Convolution)**  
   - 풀링 레이어를 사용하지 않고, 컨볼루션 연산에서 **Stride**를 크게 설정해 **다운샘플링 효과**를 얻는 방식  
   - 학습 가능한 필터를 통해 정보를 압축하므로, 단순 풀링보다 **정보 손실을 줄일 수 있음** [^4].

3. **어텐션 기반 풀링(Attention-based Pooling)**  
   - 풀링 시, 단순 최댓값/평균 대신 **학습된 어텐션 가중치**를 부여해, 영역 내 **중요 픽셀**에 더 큰 비중을 두는 기법  
   - 중요한 특징은 강조하고, 불필요한 정보는 축소하여 **유연한 다운샘플링**이 가능

4. **하이브리드/적응형 풀링(Adaptive Pooling)**  
   - 맥스와 평균을 혼합하거나, Pooling 방식을 **학습**으로 결정하는 기법  
   - 예: PyTorch의 `AdaptiveAvgPool2d` 등은 임의의 입력을 고정된 크기로 풀링해주며, 다양한 CNN 구조에서 활용

---

# 최신 연구 동향

1. **Fractional Max-Pooling**  
   - 고정된 커널 대신, 확률적 방식으로 풀링 영역을 결정하여 **다양한 스케일**의 정보를 학습 [^1].
2. **Spatial Pyramid Pooling (SPP)**  
   - 입력을 서로 다른 크기의 공간 그리드(Spatial Pyramid)로 나누어 풀링을 수행, **다중 스케일 정보**를 한 번에 추출 [^3].
3. **All Convolutional Net**  
   - **전부 컨볼루션**만으로 구성된 네트워크를 제안하여, 풀링 없이 **Stride**를 통해 다운샘플링 효과를 내는 방식을 사용 [^4].
4. **Global Pooling + Fully Convolutional**  
   - Global Pooling과 완전 컨볼루션 구조를 결합해, **입력 크기에 관계없이** 유연하게 적용 가능 (예: Network in Network, NiN [^2]).

최근 연구들은 단순히 **해상도 축소** 역할을 넘어, **특징 선택(Feature Selection)**이나 **정규화(Regularization)** 효과를 극대화할 수 있는 풀링 기법을 지속적으로 모색하고 있습니다. 특히 **어텐션 메커니즘(Attention)**이나 **학습 가능한 가중치**를 결합하여, **데이터 특성에 맞춰 동적으로 정보 요약**을 수행하는 방향이 활발하게 연구되고 있습니다.

---

## 참고문헌

[^1]: Graham, B. (2014). *Fractional max-pooling*. **arXiv preprint arXiv:1412.6071**.  
[^2]: Lin, M., Chen, Q., & Yan, S. (2013). *Network in network*. **arXiv preprint arXiv:1312.4400**.  
[^3]: He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Spatial pyramid pooling in deep convolutional networks for visual recognition*. **IEEE TPAMI**, 37(9), 1904-1916.  
[^4]: Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). *Striving for simplicity: The all convolutional net*. **arXiv preprint arXiv:1412.6806**.  
[^5]: Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2021). **Dive into deep learning**. *arXiv preprint arXiv:2106.11342*.  
[^6]: Murphy, K. P. (2012). **Machine learning: a probabilistic perspective**. MIT press.

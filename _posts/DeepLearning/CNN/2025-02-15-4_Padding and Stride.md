---
published: false
title: "[CNN] (4) 패딩(Padding)과 스트라이드(Stride)"
description: "CNN에서 패딩과 스트라이드가 Feature Map의 크기와 학습에 미치는 영향"
header:
  teaser: /assets/images/padding_stride.png
  og_image: /assets/images/padding_stride.png
  image_description: "Padding and Stride in CNN"
date: 2025-02-15
last_modified_at: 2025-02-15 21:00:00
toc: true
toc_sticky: true
use_math: true
categories:
  - CNN
tags:
  - DeepLearning
  - CNN
  - Padding
  - Stride
---

이번 포스팅에서는 컨볼루션 레이어에서 패딩(Padding)과 스트라이드(Stride)에 대해서 다뤄보겠습니다.

## CNN 시리즈
1. [Convolutional Neural Networks 개요]({{ site.url }}{{ site.baseurl }}/cnn/1_CNN-Basics/)
2. [CNN의 역사와 발전 과정, 주요 모델들]({{ site.url }}{{ site.baseurl }}/cnn/2_CNN-History/)
3. [컨볼루션 레이어(Convolutional Layer)]({{ site.url }}{{ site.baseurl }}/cnn/3_Convolution-Layer/)
<!-- 4. [패딩(Padding)과 스트라이드(Stride)]({{ site.url }}{{ site.baseurl }}/cnn/4_Padding-and-Stride/) -->
<!-- 5. [풀링 레이어(Pooling Layer)]({{ site.url }}{{ site.baseurl }}/cnn/5_Pooling-Layers/) -->
<!-- 6. [다중 채널(Multiple Channels)]({{ site.url }}{{ site.baseurl }}/cnn/6_Multiple-Channels/) -->
<!-- 7. [컨볼루션 레이어 쌓기(Stacked Convolutional Layers)]({{ site.url }}{{ site.baseurl }}/cnn/7_Stacked-Convolution-Layers/) -->

---

## 1. 내용

### 1.1 패딩(Padding)과 스트라이드(Stride)의 정의 및 수학적 표현
CNN에서의 **Padding**은 입력 데이터의 경계(Boundary)에 여분의 픽셀(주로 0, 즉 **Zero Padding**)을 추가하여, 컨볼루션(Convolution) 연산 시 **경계 정보를 보존**하고 출력(Feature Map)의 크기를 조절하는 기법입니다.  
반면, **Stride**는 컨볼루션 필터가 입력 위를 이동하는 간격을 의미하며, 이를 통해 **출력 해상도**가 조절됩니다.

2D 컨볼루션의 출력 크기는 다음 공식으로 계산됩니다:

\[
\text{Out} = \left\lfloor\frac{N + 2P - K}{S}\right\rfloor + 1
\]

여기서,
- \( N \): 입력의 크기 (높이 또는 너비)
- \( K \): 필터(커널)의 크기
- \( P \): 패딩 크기
- \( S \): 스트라이드

해당 식은 **Valid Convolution**(패딩이 없는 경우)와 **Same Convolution**(출력 크기가 입력 크기와 동일해지도록 패딩을 추가하는 경우)에서의 출력 크기를 모두 포함합니다.  
(자세한 설명은 [^1][^6] 참조)

---

### 1.2 패딩 유형: SAME vs VALID
- **SAME Padding**  
  출력 크기를 입력과 동일하게 만들기 위한 패딩 방식입니다. 예를 들어, 3×3 필터라면 입력의 양쪽에 1픽셀씩 Zero Padding(즉, \(P=1\))을 적용하고 Stride를 1로 설정하여 출력의 높이·너비가 입력과 동일하도록 할 수 있습니다.

- **VALID Padding**  
  패딩을 적용하지 않는 방식입니다. 필터가 입력의 “유효한(valid)” 영역에서만 연산을 수행하기 때문에 출력 크기가 줄어듭니다. 예를 들어, 32×32 입력에 5×5 필터를 VALID로 적용하면 출력은 28×28이 됩니다 [^3].  

주로 SAME Padding은 **경계 정보를 최대한 보존**하고 싶을 때, VALID Padding은 **출력 크기 축소**나 **계산량 감소**가 필요한 상황에서 자주 활용됩니다. (자세한 논의는 [^5][^6] 참조)

---

### 1.3 스트라이드(Stride)의 영향
Stride는 컨볼루션 필터가 이동하는 간격을 결정하여, **출력 Feature Map의 공간 해상도**에 직접적인 영향을 미칩니다.

- **Stride = 1**  
  필터가 한 칸씩 이동합니다. 상대적으로 **세밀한 특징**을 추출할 수 있고, 출력 해상도도 **높게 유지**됩니다.

- **Stride = S > 1**  
  필터가 여러 칸씩 건너뛰며 이동하므로, **출력 크기가 대략 \(1/S\) 비율**로 축소됩니다.  
  - 계산량이 줄어들어 **효율성**이 높아질 수 있으나,
  - 큰 Stride 값은 경계 사이 픽셀을 건너뛰어 **세밀한 정보 손실**을 유발할 수 있습니다.

이러한 개념은 **Downsampling**으로도 해석할 수 있는데, Stride가 커질수록 공간 해상도가 줄어드는 효과가 발생합니다 [^1][^2][^5].

---

### 1.4 패딩과 스트라이드 조합의 모델 성능에 미치는 영향
- **정보 보존과 경계 활용**  
  SAME Padding을 사용하면 모든 입력 픽셀이 출력 계산에 영향을 주어 **경계 정보**도 학습에 반영됩니다.  
- **계산 효율성**  
  Stride를 늘리면 출력 크기가 줄어 연산량이 감소합니다. 특히 입력 이미지가 클 때, 초기 레이어에서 Stride 2 등을 적용해 점진적으로 Downsampling을 수행함으로써 **계산 부담**을 줄일 수 있습니다.
- **실제 사례**  
  - ResNet(He et al.)과 같은 현대적 구조에서는 초기 레이어에서 SAME Padding을 통해 출력 크기를 유지하면서 학습을 안정화한 뒤, 필요 시 Stride 2를 이용해 Downsampling을 수행합니다 [^2].  
  - MobileNet(Howard et al.)은 **Depthwise Convolution**과 적절한 Stride 설정을 통해 **연산량을 크게 줄이면서도** 휴대폰 환경 등에서 실용적인 정확도를 달성합니다 [^4].  
  - VGG(Simonyan & Zisserman)는 VALID 방식으로 빠르게 깊게 쌓는 전략을 취했으며, 이후 Pooling 레이어와 결합하여 Feature Map 크기를 점진적으로 줄여나갑니다 [^3].

---

### 1.5 최신 연구 동향 및 적용 사례
최근에는 **Padding**과 **Stride** 설정이 CNN의 학습 및 일반화(Generalization)에 미치는 영향을 정량적으로 분석하고, 이를 최적화하는 방법이 다수 제시되고 있습니다.

- *Dumoulin & Visin (2016)* [^1]  
  CNN에서의 컨볼루션 연산 방식 및 수학적 구조를 체계적으로 정리하며, Padding과 Stride가 모델 성능과 연산 효율성에 미치는 중요성을 강조합니다.

- *He et al. (2016)*의 ResNet [^2]  
  초기 단계에서 SAME Padding을 적용하고, Residual Block 내부 혹은 특정 위치에서 Stride 2로 Downsampling을 수행하여 **깊은 네트워크의 안정적 학습**과 높은 성능을 달성하였습니다.

- *Howard et al. (2017)*의 MobileNet [^4]  
  Depthwise + Pointwise Convolution을 적용하며, **Stride 조절**을 통해 **경량화와 정확도**를 균형 있게 맞춘 대표적인 모델입니다.

- *Zhang et al. (2021)*의 *Dive into Deep Learning* [^5]  
  Padding과 Stride가 **Output Feature Map**의 크기 및 **계산 효율**에 미치는 영향을 다양한 예제 코드로 설명합니다. 큰 커널을 사용하는 경우, 적절한 Stride와 Padding을 함께 설정해 **실용적인 해상도**와 **계산량**을 맞추는 전략을 소개합니다.

- *Murphy (2012)*의 *Machine Learning: A Probabilistic Perspective* [^6]  
  CNN에서 **Boundary Condition**을 어떻게 처리하느냐(Valid vs. Same vs. Zero Padding 등)에 따라 **특징 추출 범위**와 **출력 크기**가 크게 달라짐을 수학적으로 정리하였습니다. 또한, **Stride Convolution**이 갖는 Downsampling 효과와 이로 인한 **특징 중복 제거**(redundancy reduction) 가능성을 강조합니다.

이처럼 CNN 아키텍처 설계 시, Padding과 Stride의 설정은 단순한 하이퍼파라미터 이상의 의미를 가지며, **모델의 학습 안정성**, **표현력**, **계산 효율성** 등에 직결되는 중요한 요소입니다.

---

## 참고문헌

[^1]: Dumoulin, V., & Visin, F. (2016). **A guide to convolution arithmetic for deep learning**. arXiv preprint arXiv:1603.07285.  
[^2]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep residual learning for image recognition**. CVPR, 770-778.  
[^3]: Simonyan, K., & Zisserman, A. (2014). **Very deep convolutional networks for large-scale image recognition**. arXiv preprint arXiv:1409.1556.  
[^4]: Howard, A. G., Zhu, M., Chen, B., et al. (2017). **Mobilenets: Efficient convolutional neural networks for mobile vision applications**. arXiv preprint arXiv:1704.04861.  
[^5]: Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2021). **Dive into deep learning**. arXiv preprint arXiv:2106.11342.  
[^6]: Murphy, K. P. (2012). **Machine learning: a probabilistic perspective**. MIT press.

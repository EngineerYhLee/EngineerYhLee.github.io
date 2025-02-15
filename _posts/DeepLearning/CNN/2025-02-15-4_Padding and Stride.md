---
published: false
title: "(4) CNN의 패딩과 스트라이드 이해와 응용"
description: "CNN에서 패딩과 스트라이드가 특징 맵의 크기와 학습에 미치는 영향"
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

## 1. 내용

### 1.1 패딩(Padding)과 스트라이드(Stride)의 정의 및 수학적 표현
CNN에서의 **패딩**은 입력 데이터의 경계에 여분의 픽셀(보통 0으로 채움)을 추가하여 컨볼루션 연산 시 경계 정보를 보존하고 출력(feature map)의 크기를 조절하는 기법입니다. 반면, **스트라이드**는 컨볼루션 필터가 입력 위를 이동하는 간격을 의미하며, 이로 인해 출력 해상도가 조절됩니다.

2D 컨볼루션의 출력 크기는 다음 공식으로 계산됩니다:

$$
\text{Out} = \left\lfloor\frac{N + 2P - K}{S}\right\rfloor + 1
$$

여기서,
- \(N\)은 입력의 크기 (높이 또는 너비),
- \(K\)는 필터의 크기,
- \(P\)는 패딩 크기,
- \(S\)는 스트라이드입니다 [^1].

### 1.2 패딩 유형: SAME vs VALID
- **SAME 패딩:**  
  출력 크기를 입력과 동일하게 유지하도록 패딩을 적용합니다. 예를 들어, 3×3 필터의 경우 양쪽에 1픽셀씩 0을 추가하여 \(P=1\)로 설정하면, 스트라이드 1에서는 출력 크기가 입력과 동일하게 됩니다.

- **VALID 패딩:**  
  패딩을 적용하지 않고, 필터가 입력의 “유효한” 영역에서만 연산되도록 하여 출력 크기가 줄어듭니다. 예를 들어, 32×32 입력에 5×5 필터를 VALID 패딩으로 적용하면 출력은 28×28로 감소합니다 [^3].

두 방식은 경계 정보 활용 방식과 계산량 측면에서 차이가 있으며, 모델의 목적에 따라 적절한 방식을 선택합니다.

### 1.3 스트라이드의 영향
스트라이드는 컨볼루션 필터가 이동하는 간격을 결정하여, 출력 특징 맵의 **공간 해상도**에 직접적인 영향을 미칩니다.
- **스트라이드 1:**  
  필터가 한 칸씩 이동하여 세밀한 특징을 추출하고, 출력 해상도가 높게 유지됩니다.
- **스트라이드 \(S>1\):**  
  필터가 여러 칸씩 건너뛰며 이동하므로 출력 크기가 대략 \(1/S\)로 축소됩니다. 이는 계산량 감소와 함께 더 넓은 문맥 정보를 한 번에 반영할 수 있으나, 너무 큰 스트라이드는 세밀한 정보 손실로 이어질 수 있습니다 [^1][^2].

### 1.4 패딩과 스트라이드 조합의 모델 성능에 미치는 영향
패딩과 스트라이드의 적절한 조합은 CNN의 성능에 결정적인 역할을 합니다.
- **정보 보존:**  
  SAME 패딩은 모든 입력 픽셀이 출력에 반영되도록 하여 경계 정보를 최대한 보존합니다.
- **계산 효율성:**  
  적절한 스트라이드 적용은 다운샘플링을 통해 연산량을 줄여 효율성을 높입니다.
- **실제 사례:**  
  예를 들어, ResNet과 같은 최신 모델에서는 초기 컨볼루션에 SAME 패딩을 적용하여 출력 크기를 유지하고, 다운샘플링이 필요한 부분에서는 스트라이드 2를 사용하여 점진적으로 해상도를 낮춥니다. 또한, 일부 연구에서는 제로 패딩 대신 평균 또는 반사 패딩(reflection padding)을 사용해 경계 효과를 줄이고 성능을 개선한 사례도 있습니다 [^2][^3].

### 1.5 최신 연구 동향 및 적용 사례
최근 연구들은 **패딩**과 **스트라이드**의 설정이 CNN의 내부 표현과 일반화에 미치는 영향을 정량적으로 분석하고, 이를 최적화하는 방법을 제시하고 있습니다.
- *Dumoulin & Visin (2016)*은 컨볼루션 연산의 수학적 구조와 함께, 패딩과 스트라이드의 역할을 체계적으로 정리하여 CNN 설계 시 고려해야 할 주요 요소임을 보여주었습니다 [^1].
- *He et al. (2016)*의 ResNet은 적절한 패딩과 스트라이드 조합을 통해 깊은 네트워크에서도 효과적인 다운샘플링과 특징 학습을 가능하게 했으며, 이러한 전략은 모델의 안정성과 성능 향상에 기여하였습니다 [^2].
- *Howard et al. (2017)*의 MobileNet은 경량 모델을 위해 스트라이드를 조정하고, Depthwise Convolution과 함께 효율적인 구조를 구현하여 모바일 환경에서 높은 성능을 보여줍니다 [^4].

이처럼 CNN 아키텍처 설계 시, 패딩과 스트라이드의 설정은 단순한 하이퍼파라미터 이상의 의미를 가지며, 모델의 학습, 표현력, 계산 효율성에 큰 영향을 미칩니다.

---

# 참고문헌

[^1]: Dumoulin, V., & Visin, F. (2016). *A guide to convolution arithmetic for deep learning*. **arXiv preprint arXiv:1603.07285**.
[^2]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition*. **CVPR**, 770-778.
[^3]: Simonyan, K., & Zisserman, A. (2014). *Very deep convolutional networks for large-scale image recognition*. **arXiv preprint arXiv:1409.1556**.
[^4]: Howard, A. G., Zhu, M., Chen, B., et al. (2017). *Mobilenets: Efficient convolutional neural networks for mobile vision applications*. **arXiv preprint arXiv:1704.04861**.

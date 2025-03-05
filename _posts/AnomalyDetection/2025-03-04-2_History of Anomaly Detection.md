---
published: true
title: "[Anomaly Detection] 이상 탐지의 개념과 역사"
description: "이상탐지(Anomaly Detection)의 개념과 역사에 대한 전반적인 내용을 설명합니다."
date: 2025-03-04
last_modified_at: 2025-03-04
toc: true
toc_sticky: true
use_math: true
categories:
  - AnomalyDetectionBasics
tags:
  - AnomalyDetection
  - History
---

## 1. 이상 탐지(Anomaly Detection)의 개념
이상 탐지(Anomaly Detection)는 데이터에서 **정상적인 패턴(Normal Pattern)**과는 크게 다른 **이상치(Outlier)**를 식별하는 과정을 의미합니다[^1]. 정상 패턴이란 대다수 데이터가 따르는 규칙적이고 예측 가능한 행동이나 분포를 지칭하며, 이상치는 드물게 나타나서 정상 범주와 현저히 벗어난 관측값·이벤트를 말합니다[^1]. 예컨대 Hawkins(1980)는 이상치를 “다른 관측치들과 현저히 동떨어져 있어서 **별도의 메커니즘**에 의해 생성된 것이라 의심되는 관측값”이라 정의하였는데, 이는 정상 행동의 개념에 부합하지 않고 데이터의 나머지 부분과 **일관되지 않은 패턴**을 나타낸다고 볼 수 있습니다[^1].

이상 탐지가 중요한 이유는, 이상치가 단순 노이즈가 아니라 종종 **중요한 의미**나 **위험 신호**를 담고 있기 때문입니다. 실제로 금융, 보안, 제조, 의료 등 다양한 산업 및 연구 분야에서 이상 탐지는 핵심 역할을 수행합니다[^1]:

- **금융(Finance)**: 신용카드 사기 탐지나 이상 거래 감지를 통해 금융 사기를 차단하고 리스크 관리에 기여.
- **보안(Security)**: 네트워크 Intrusion Detection(IDS) 및 사이버보안에서 비정상 트래픽 또는 수상한 행동을 조기에 발견.
- **제조(Manufacturing)**: 공정 데이터나 설비 센서 정보를 분석하여 결함 신호나 고장을 예측(Predictive Maintenance)해 품질 이상을 조기 탐지.
- **의료(Healthcare)**: 환자의 이상 징후, 병변을 조기에 파악하거나 의료 기기 오류를 감지함으로써 안전성과 진단 정확도 향상.
- **기타**: 영상 결함 검출, 센서 네트워크 이상 이벤트, 환경 모니터링, IT 시스템 장애 조짐 탐지 등 광범위한 적용.

이처럼 이상 탐지는 다양한 영역에서 “정상 패턴에서 벗어나는 특이 신호”를 찾아내어 **심각한 문제를 예방하거나 빠르게 대응**할 수 있도록 돕습니다[^1].

---

## 2. 이상 탐지(Anomaly Detection)의 역사

이상 탐지 기술은 **통계적 방법**에서 출발해, 이후 **Machine Learning** 기반으로 발전하고, 최근에는 **Deep Learning** 기법과 결합하여 성능과 적용 범위를 넓혀 왔습니다. 시대적·방법론적 흐름을 간단히 살펴보면 다음과 같습니다.

### 2.1 통계 기반 전통적 기법

초기에 이상치는 주로 통계학적 분포 가정에 따라 판별되었습니다. 예를 들어 **정규분포(Gaussian)**를 가정하고 평균·분산 범위를 벗어나는 점을 이상치로 보는 **z-score** 방식이나, 특정 통계 검정(Grubbs, Tukey IQR 등)을 통해 극단값을 제거하는 방법 등이 대표적이었습니다[^1]. 또한 **Statistical Process Control(SPC)**에서 사용된 관리도(Control Chart) 기법은 공정 데이터가 제어 한계를 벗어나는지를 확인해 실시간으로 이상 여부를 감지하는 등 산업계 품질관리 분야에 일찍부터 활용되었습니다[^2].  

이 같은 전통적 방법은 **이론적으로 간단**하고 **해석이 쉬우며**, 적은 데이터로도 적용 가능한 장점이 있으나, 복잡하고 다차원적인 분포나 다양한 이상 패턴에는 대응이 어려운 단점이 있었습니다.

### 2.2 Machine Learning 기반 접근

1990년대 후반~2000년대에는 기존 통계 방법의 한계를 극복하기 위해 **Machine Learning** 기법들이 등장했습니다[^3]. 대표적으로 **One-Class SVM**은 정상 데이터만으로 결정 경계(Decision Boundary)를 형성해 그 밖의 점을 이상치로 처리하는 방식으로, 네트워크 Intrusion Detection 등 다양한 분야에 적용되어 주목받았습니다[^4].  

군집(Clustering) 기반 접근은 k-means나 밀도 기반(LOF 등) 알고리즘을 사용해 밀도가 낮은 점을 이상치로 판별했고, **Isolation Forest**는 분할 기반 앙상블(Tree Ensemble)로 이상치를 빠르게 격리하여 대규모 데이터 처리에 효과적이었습니다[^1][^5]. 이 시기 기법들은 대부분 **얕은(Shallow) 모델**에 의존했으며, 대체로 통계적 가정보다는 **데이터의 구조**나 **밀도**, **거리** 등을 통해 이상값을 정의하는 방식을 택했습니다.

### 2.3 Deep Learning 기반 이상 탐지 기법

2010년대 중반 이후 **Deep Learning**이 부상하면서, 복잡한 비선형 패턴을 학습할 수 있는 **신경망**이 이상 탐지에도 도입되었습니다[^3].  
- **Autoencoder**: 정상 데이터로 학습해 **Reconstruction Error**를 기준으로 이상치를 구분. 비선형 Autoencoder나 Variational Autoencoder(VAE)는 이미지·시계열·신호 등에서 우수한 성능을 보임.  
- **Generative Adversarial Network (GAN)**: 정상 분포를 학습한 생성자가 새로운 샘플을 재현하기 어렵다면 이상치로 판별(AnoGAN 등).  
- **Convolution Neural Network(CNN)/Recurrent Neural Network(RNN)**: 영상 이상 탐지나 시계열 패턴 인식에 특화. 시계열 데이터의 경우 LSTM, GRU 등을 사용해 시간적 이상 신호를 포착.  

이 같은 딥러닝 접근은 고차원·대규모 데이터에서 뛰어난 결과를 제공하면서도, **훈련 안정성**, **모델 해석** 등의 과제가 함께 제기되었습니다[^3].

---

## 3. 최신 연구 동향 (2020년 이후)

### 3.1 Self-Supervised/비지도 학습

레이블 부족 문제 해결을 위해 **Self-Supervised Learning** 기법이 이상 탐지에도 적극 도입되고 있습니다. 정상 데이터에 인위적 변형(예: Patch 삽입, 마스킹)을 주어 원본 vs 변형을 구별하도록 학습하면, 모델이 **정상 패턴**을 깊이 학습해 새로운 이상에도 강건해지는 방식입니다[^6]. 또한 대규모 **Pretrained Model**(예: ImageNet 사전학습)로부터 특성을 가져와 적은 데이터로도 효과적으로 이상을 감지하는 연구가 늘고 있습니다.

### 3.2 Transformer 기반 및 사전학습 모델

시계열 이상 탐지에서는 **Transformer** 구조(예: Anomaly Transformer)가 시점 간 **장기 의존성(Long-Range Dependency)**을 학습해 높은 정확도를 내고 있습니다[^7]. 또한 Vision-Language 모델(CLIP 등)을 활용해 **제로샷** 이상 탐지나 **멀티모달** 접근을 시도하는 등의 연구도 활발합니다[^8].

### 3.3 Diffusion Model 활용

Diffusion Model은 생성 모델 분야의 최신 흐름으로, 노이즈 제거 과정을 통해 **데이터 분포**를 정확히 학습한다는 장점이 있습니다. 이를 이용해 입력 데이터와 생성 분포 간 **오차** 또는 **확률 밀도**를 측정해 이상값을 추론하는 시도가 늘고 있습니다[^9]. GAN 대비 학습 안정성이 높아, 이미지·시계열 전반에 적용 가능하다는 평가가 있습니다.

### 3.4 연구 전망

최근에는 방대한 벤치마크(예: ADBench 등)와 함께 **다양한 알고리즘**을 종합 비교하며, 이상 탐지의 성능·특성·적용 한계를 체계적으로 분석하려는 시도가 많아지고 있습니다. 미래에는 **스트리밍·고차원·멀티모달** 데이터에서 **실시간 이상 탐지**가 중요해질 것으로 보이며, 모델의 **설명 가능성**(Why is it anomalous?), **도메인 지식과의 결합**, **오탐/미탐 최소화 전략** 등이 연구 과제로 부상하고 있습니다[^3].

---

## 4. 참고문헌

[^1]: Anomaly detection - Wikipedia. <https://en.wikipedia.org/wiki/Anomaly_detection>  
[^2]: Shewhart Control Charts - ResearchGate. <https://www.researchgate.net/publication/225775194_Statistical_Process_Control_using_Shewhart_Control_Charts_with_Supplementary_Runs_Rules>  
[^3]: Pang, G., Shen, C., Cao, L., & van den Hengel, A. (2021). *Deep Learning for Anomaly Detection: A Review*. **ACM Computing Surveys**, 54(2), 38:1–38:38. [[2007.02500] Deep Learning for Anomaly Detection: A Review](https://arxiv.org/abs/2007.02500)  
[^4]: Schölkopf, B., Platt, J., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). *Estimating the support of a high-dimensional distribution*. **Neural Computation**, 13(7), 1443–1471.  
[^5]: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). *Isolation Forest*. **2008 IEEE International Conference on Data Mining**, 413–422.  
[^6]: Li, C.-L., Sohn, K., Yoon, J., & Pfister, T. (2021). *CutPaste: Self-Supervised Learning for Anomaly Detection and Localization*. **CVPR**, 9664–9674.  
[^7]: Xu, J., Wu, H., Wang, J., & Long, M. (2022). *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy*. **ICLR**.  
[^8]: Jeong, J., Zou, Y., Kim, T., & et al. (2023). *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation*. **CVPR**.  
[^9]: Liu, Z., Zhou, J. P., & Weinberger, K. Q. (2024). *Leveraging Diffusion Models for Unsupervised Out-of-Distribution Detection on Image Manifold*. **Frontiers in Artificial Intelligence**, 7.


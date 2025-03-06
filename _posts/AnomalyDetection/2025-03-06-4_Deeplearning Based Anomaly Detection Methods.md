---
published: true
title: "[Anomaly Detection] 딥러닝 기반 이상 탐지 기법"
description: "딥러닝 기반 이상 탐지 기법을 정리하고 설명합니다."
date: 2025-03-06
last_modified_at: 2025-03-06
toc: true
toc_sticky: true
use_math: true
categories:
  - AnomalyDetectionBasics
tags:
  - AnomalyDetection
  - Deeplearning
---

## 1. 딥러닝의 이상 탐지 적용 방식
### 1.1 전통적 기법 vs 딥러닝
이상 탐지(Anomaly Detection)는 오래전부터 통계 기반, 거리 기반, 밀도 기반, SVM, Isolation Forest 등 다양한 전통적 기법으로 연구되어 왔습니다. 그러나 데이터 규모와 복잡도가 커지면서 딥러닝(Deep Learning)이 전통 방법보다 뛰어난 성능을 보이게 되었고, 이를 **deep anomaly detection**이라고 부릅니다[^1]. 딥러닝은 복잡한 고차원 데이터에서 **표현(Representation)**을 자동으로 학습하여 이상치를 더 정교하게 식별할 수 있습니다.

### 1.2 지도/준지도/비지도 학습의 차이
- **지도 학습(Supervised Learning)**: 정상/이상 레이블을 학습에 사용. 하지만 이상 레이블 확보가 어려워 실제 적용 사례는 적습니다[^1].
- **준지도 학습(Semi-supervised Learning)**: 정상 데이터만으로 모델을 학습하고, 추후 재구성 오류(Reconstruction Error) 등의 지표로 이상 여부를 판별합니다. **Autoencoder**(AE) 방식이 대표적입니다.
- **비지도 학습(Unsupervised Learning)**: 레이블 없이 정상/이상 패턴을 모델링합니다. **VAE**, **GAN**, **Clustering** 기반 등으로 확장되어 연구되고 있습니다. 레이블 없이도 가능하나 이상 데이터 비율이 높으면 성능이 저하될 수 있습니다.

## 2. 대표적인 딥러닝 기반 이상 탐지 모델 개요

딥러닝을 활용한 이상 탐지 모델은 사용되는 **신경망 구조**와 **학습 방식**에 따라 여러 계열로 나뉩니다[^1]. 대표적으로 Autoencoder, GAN, CNN/RNN, Transformer, Self-Supervised Learning 기반 방법 등이 있습니다.

### 2.1 Autoencoder 계열
- **Basic Autoencoder (AE)**  
  정상 데이터만으로 인코더-디코더 구조를 학습하고, 추후 입력을 재구성하는 과정에서 오류가 크면 이상으로 간주합니다[^1].  
- **Variational Autoencoder (VAE)**, **Denoising AE**, **Sparse AE** 등 다양한 변형 모델이 제안되어 왔으며, 비슷한 아이디어로 **LSTM-AE**는 시계열 데이터를 재구성해 이상을 탐지합니다.
- **Memory-Augmented Autoencoder (MemAE)**  
  AE에 **Memory Module**을 추가하여, 정상 프로토타입 패턴만을 메모리에 저장하고 이를 이용해 복원합니다. 이상 데이터는 정상 패턴으로 복원되지 않아 재구성 오류가 커집니다[^2].

### 2.2 GAN 기반 모델
- **AnoGAN**  
  정상 데이터만으로 **Generative Adversarial Network (GAN)**을 학습한 뒤, 입력 샘플이 학습된 정상 분포에서 얼마나 벗어나는지로 이상 정도를 측정합니다[^3].  
- **f-AnoGAN**  
  AnoGAN의 잠재 벡터 최적화 단계를 개선하기 위해 Encoder 네트워크를 추가한 구조로, **inference 속도**를 높였습니다[^4].  
- **GANomaly**  
  Generator 내부에 인코더-디코더-인코더를 결합해, 정상 데이터 분포를 준지도 학습으로 효과적으로 학습합니다. 입력과 재생성 잠재벡터의 거리를 이상도(Anomaly Score)로 사용합니다[^5].

### 2.3 CNN/RNN 기반 모델
CNN(Convolutional Neural Network)과 RNN(Recurrent Neural Network)은 각각 공간적·시간적 패턴 학습에 강점을 지니며, 시계열(Time Series), 로그(Log), 네트워크 패킷 등 순차 데이터에서 이상 탐지에 자주 쓰입니다.  
- **LSTM(Encoder-Decoder)**: 정상 시계열을 재구성하도록 학습한 후, 재구성 오류를 이상으로 판단[^6].  
- **CNN-LSTM** 혼합: CNN으로 지역적 특징을 추출하고 LSTM으로 장기 의존성을 모델링해 시계열 이상을 탐지.  
- **DeepLog**: RNN으로 다음 로그 이벤트를 예측하고, 예측 불가능한 이벤트가 등장하면 이상으로 간주[^7].

### 2.4 Transformer 기반 모델
자기어텐션(Self-Attention)으로 시계열 전역적 상관관계를 학습하는 **Transformer**는 긴 시계열이나 로그에서도 강력함을 보입니다.  
- **Anomaly Transformer**: 각 시점의 어텐션 패턴을 이용해 **Association Discrepancy**를 계산함으로써, 정상 시계열과 이상 시계열을 구분하는 기법을 제안합니다[^8]. 긴 시계열에서도 우수한 성능을 입증했습니다.

### 2.5 Self-Supervised Learning 기반 모델
레이블 없이 **자기지도(Self-Supervised)** 과제를 부여하여 정상 패턴을 학습한 뒤, 과제 수행 결과로 이상 여부를 판단합니다.  
- **GeoTrans**: 정상 이미지에 무작위 기하학 변환(회전·뒤집기 등)을 적용하고 어떤 변환이었는지 맞추는 분류기를 학습해, 이상 데이터는 변환 분류에 실패하는 정도로 구분[^9].  
- **대조 학습(Contrastive Learning)**: 같은 샘플에서 증강된 두 뷰를 가깝게, 다른 샘플에서 온 뷰는 멀게 학습해 특징 표현을 강화. TS-TCC 같은 기법을 활용하여 소량 라벨 환경에서도 높은 성능 달성[^10].

## 3. 모델별 동작 원리 및 적용 사례

### 3.1 Autoencoder 기반 이상 탐지
**재구성 오류(Reconstruction Error)**를 기반으로 이상을 탐지합니다. 정상 데이터로 AE를 학습하면 정상 패턴을 복원 능력이 높아지고, 이상 입력에서는 오류가 커집니다[^1]. 제조 공정 결함 탐지(MVTec AD[^11]), 네트워크 침입 탐지(KDD Cup 99, NSL-KDD[^12][^13]) 등 다양한 영역에서 간단하면서도 효과적인 방법으로 널리 사용됩니다. 단, AE가 이상 패턴까지 재현해버리는 문제를 해결하기 위해 MemAE 등 여러 변형이 제안되었습니다[^2].

### 3.2 GAN 기반 이상 탐지
**생성 모델**이 정상 분포를 학습해두고, 입력이 그 분포에 맞지 않으면 이상으로 판단합니다.  
- **AnoGAN/f-AnoGAN**은 정상 데이터로 학습한 GAN에서, 입력을 가장 가깝게 생성하는 잠재 벡터를 찾은 뒤 (또는 Encoder로 직접 추정), 생성된 정상 이미지와의 차이·판별자 특징 차이로 이상도를 구합니다[^3][^4].  
- **GANomaly**는 Generator 구조에 Encoder-Decoder-Encoder를 탑재해, 정상 입력과 출력의 잠재 표현 불일치가 큰 경우를 이상으로 간주합니다[^5].

### 3.3 CNN/RNN 기반 모델
**시계열, 로그, 이벤트 스트림** 등 순차 데이터에 주로 쓰입니다.  
- **예측 기반**: LSTM, Transformer 등으로 다음 시점이나 다음 이벤트를 예측하고, 실제값과 차이가 크면 이상으로 판단 (예: DeepLog, 서버 성능 지표 모니터링)[^7].  
- **재구성 기반**: LSTM Encoder-Decoder(Seq2Seq)로 정상 시계열을 복원하고, 오류가 큰 구간을 이상으로 간주[^6].  
네트워크 침입 탐지(NSL-KDD 등)나 시스템 로그(HDFS 로그), CCTV 비디오 이상 행동 탐지에도 다수 적용됩니다.

### 3.4 Transformer 기반 모델
자기어텐션으로 시계열 전역 의존성을 학습하는 강점을 활용해, 긴 시계열에서도 강력한 이상 탐지를 수행합니다.  
- **Anomaly Transformer**: 각 시점의 어텐션 분포가 정상과 달라지는 정도(Association Discrepancy)를 측정하고, Minimax 최적화로 정상·이상 분포를 분리[^8]. NASA 위성 로그, 산업 센서 등 다양한 벤치마크에서 SOTA급 성능을 보였습니다.

### 3.5 Self-Supervised Learning 기반 모델
레이블이 거의 없는 현실 환경에서, 모델이 스스로 만든 **가짜 태스크**를 학습하여 정상 특징을 깊이 이해한 뒤 이상을 판별합니다.  
- **기하학 변환 분류(GeoTrans)**: 정상 데이터에 여러 변환을 적용하고, 어떤 변환인지 맞추도록 분류기를 학습. 이상 데이터는 잘 맞추지 못해 확률분포가 무너짐[^9].  
- **TS-TCC**: 증강된 시계열 뷰를 대조 학습해, 적은 라벨로도 정확한 이상 탐지 모델을 구성[^10].  
산업 결함(CutPaste), 자율주행 경로, 금융 거래 등 다양한 도메인에서 레이블 부족 문제를 해결하는 데 큰 역할을 합니다.

## 4. 학습 데이터 및 평가 방법

### 4.1 학습 데이터 구성
대부분 **정상 데이터**가 풍부하고 이상 레이블이 희귀하므로, 준지도·비지도 접근이 일반적입니다.  
- **KDD Cup 99**, **NSL-KDD**: 침입 탐지용 표준 벤치마크.  
- **MVTec AD**: 제조 결함 이미지 데이터셋.  
- **SWaT**, **SMAP/MSL**: 산업 공정·위성 센서 시계열.  
- **NAB**, **Yahoo Webscope S5**: 일반 시계열 이상 탐지용.

### 4.2 평가 지표
- **Precision**(정밀도), **Recall**(재현율), **F1-Score**, **ROC-AUC**, **PR-AUC** 등을 주로 사용[^1].  
- 이상 탐지 특성상 정상:이상 클래스 불균형이 심하므로 **PR-AUC**가 더 의미 있는 경우도 많습니다.  
- 산업 현장에서는 **False Alarm Rate(오탐율)**, **Miss Rate(미탐율)**, **실시간성** 등도 주요 지표입니다.

## 5. 실무 적용 시 고려 사항

1. **데이터 분포 변화(Concept Drift)**  
   시간이 지나면서 정상 분포 자체가 바뀌면, 고정된 딥러닝 모델이 오탐을 많이 낼 수 있습니다. 따라서 **모델 재학습**이나 **온라인 학습** 전략을 고려해야 합니다[^1][^15].  

2. **실시간 처리 및 시스템 부하**  
   초당 대량 데이터(예: 네트워크 패킷) 환경에서 **추론 속도**가 병목이 될 수 있으므로, **경량 모델**, **에지 컴퓨팅**, **분산 처리** 등이 필요합니다.

3. **임계치(Threshold) 설정**  
   딥러닝 이상 탐지 모델은 연속적 이상 점수만 내놓으므로, 어느 지점을 기준으로 이상으로 볼지 **도메인 전문가**와 함께 정해야 합니다. Precision-Recall 트레이드오프를 고려해 다단계 필터링 등을 도입하기도 합니다.

4. **해석 가능성(XAI)**  
   딥러닝은 블랙박스 특성이 있어, 결과를 해석하기 어려운 문제가 있습니다[^16].  
   - **재구성 차이 시각화**: Autoencoder 출력과 입력 차이를 영상으로 표시해 어느 부위가 이상인지 설명.  
   - **Attention 기반 강조**: Transformer나 RNN 모델에서 어떤 시점(또는 변수)이 이상 판단에 기여했는지 가중치를 시각화.  
   - **Grad-CAM, SHAP** 등 일반적인 XAI 기법도 적용 가능.

---

## 참고문헌

[^1]: Chalapathy, R., & Chawla, S. (2019). *Deep Learning for Anomaly Detection: A Survey*. **arXiv preprint arXiv:1901.03407**.  
[^2]: Gong, D. et al. (2019). *Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder for Unsupervised Anomaly Detection*. **ICCV**.  
[^3]: Schlegl, T. et al. (2017). *Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery (AnoGAN)*. **IPMI**, arXiv:1703.05921.  
[^4]: Schlegl, T. et al. (2019). *f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks*. **Medical Image Analysis**.  
[^5]: Akcay, S., Atapour-Abarghouei, A., & Breckon, T. (2018). *GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training*. **ACCV**, arXiv:1805.06725.  
[^6]: Malhotra, P. et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection*. **ICML Workshop**, arXiv:1607.00148.  
[^7]: Du, M. et al. (2017). *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*. **ACM CCS**.  
[^8]: Xu, J. et al. (2022). *Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy*. **ICLR**, arXiv:2110.02642.  
[^9]: Golan, I., & El-Yaniv, R. (2018). *Deep Anomaly Detection Using Geometric Transformations*. **NeurIPS**, arXiv:1805.10917.  
[^10]: Eldele, E. et al. (2021). *Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC)*. **IJCAI**, arXiv:2106.14112.  
[^11]: Bergmann, P. et al. (2019). *MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. **CVPR**.  
[^12]: KDD Cup 1999 Dataset. <http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html>  
[^13]: NSL-KDD Dataset. <https://www.unb.ca/cic/datasets/nsl.html>  
[^14]: NASA SMAP/MSL Datasets. <https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/>  
[^15]: Doshi, K. et al. (2020). *Continual Learning for Anomaly Detection*. **ICPR**.  
[^16]: Yang, F. et al. (2021). *Explainable Deep One-Class Classification*. **arXiv preprint arXiv:2109.01884**.

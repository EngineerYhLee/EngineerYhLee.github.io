---
published: true
title: "[Anomaly Detection] 이상 탐지(Anomaly Detection) 개요"
description: "이상 탐지(Anomaly Detection)에 대한 기본적인 개요를 설명합니다."
date: 2025-03-04
last_modified_at: 2025-03-04

toc: true
toc_sticky: true
use_math: true
categories:
  - AnomalyDetectionBasics
tags:
  - AnomalyDetection
---

## 1. 개요
### 1.1. 이상 탐지(Anomaly Detection)란?
이상 탐지는 정상 패턴에서 크게 벗어나는 **이상치**를 식별하는 기술로, 금융 사기 방지나 시스템 장애 예측 등 다양한 영역에서 중요한 역할을 합니다[^1]. 최근 **딥러닝(Deep Learning)** 기법이 급속도로 발전하면서, 과거 전통적 방법을 뛰어넘는 높은 정확도와 적응성을 보이는 이상 탐지 모델들이 속속 등장하고 있습니다[^2].  

특히 **자기 지도 학습(Self-Supervised Learning)** 방식이 주목받고 있는데, 이는 데이터에 인위적인 변형을 가한 뒤 원본과 구별하도록 학습하여 모델이 **정상 패턴**을 보다 정교하게 습득하도록 만드는 방법입니다[^2]. 또한 시계열 분석 분야에서는 **트랜스포머(Transformer)** 기반 모델들이 복잡한 장기 의존성(Long-Range Dependency)을 다룰 수 있어, 다양한 **멀티채널 시계열** 이상 탐지에서 뛰어난 성능을 보고하고 있습니다[^3][^4].  

---

## 2. 대표 딥러닝 기법

이상 탐지 분야에서 널리 사용되는 주요 딥러닝 모델들은 아래와 같습니다:

1. **오토인코더(Autoencoder)**  
   - 정상 데이터의 **재구성 오류(Reconstruction Error)**를 이용해 이상 여부를 판별.  
   - 변분 오토인코더(VAE)나 메모리 네트워크 접목 오토인코더 등 성능을 개선한 변종들이 연구됨.  
   - 정상 데이터는 재구성 오류가 작고, 이상 데이터는 크게 나타나는 특성을 활용[^5].

2. **GAN(Generative Adversarial Network) 계열**  
   - **AnoGAN** 등 GAN 기반 접근은 정상 데이터 분포를 학습한 생성자(Generator)가, 입력 샘플과 얼마나 비슷한지를 통해 이상 유무를 판단.  
   - **GANomaly**와 같은 변형 모델들은 재구성 차이나 인코딩 차이를 분석하여 이상치를 탐지.  
   - 주로 **영상 이상 탐지**(의료, 제조 등)에 활발히 사용[^6].

3. **원클래스(One-Class) 딥러닝**  
   - 전통적 One-Class SVM, SVDD를 뉴럴넷으로 확장한 **Deep SVDD** 등이 대표적.  
   - 저차원 공간에서 단일 구로 정상 영역을 정의하고, 그 밖의 데이터를 이상으로 간주.  
   - 결정을 위한 **명시적 경계**가 형성되므로, 복잡한 이상 패턴도 어느 정도 커버 가능.

4. **시계열 모델(LSTM/RNN 등)**  
   - 연속된 시점 데이터를 입력받아 **다음 시점 예측** 또는 **재구성**을 시도, 예측 오차가 크면 이상으로 판단.  
   - LSTM, GRU 등이 사용되지만, 최근에는 장기 의존성 한계를 극복하기 위해 트랜스포머가 선호되는 추세[^4].

5. **트랜스포머(Transformer)**  
   - **Self-Attention** 기법으로 시계열·멀티채널 데이터 내 광범위한 상관관계를 학습.  
   - 예: **Anomaly Transformer**는 시계열 전체 패턴의 연관성을 학습해 재구성이나 예측 오차를 넘어서는 새로운 이상 판단 기준을 제시[^3][^4].  
   - 센서 데이터나 로그 분석, 금융 시계열 등에서 최고 수준의 성능을 달성 중.

6. **그래프 신경망(GNN)**  
   - 데이터가 **노드-엣지**로 구성된 그래프 형태일 때, **그래프 이상 탐지**를 위해 각 노드 임베딩과 구조적 관계를 함께 학습.  
   - 소셜 네트워크 사용자 이상 탐지, 통신 네트워크 트래픽 분석 등에 활용[^3].

7. **확산 모델(Diffusion Model)**  
   - **노이즈 제거** 과정을 단계적으로 진행하며 이미지를 생성하는 확산 모델을 이상 탐지에 응용.  
   - 정상 데이터 분포를 정확히 복원하고, 벗어나는 경우를 이상으로 판별하는 아이디어.  
   - 최근 의료 영상 등에서 도입 사례 보고[^3].

---

## 3. 적용 분야

1. **금융(Finance)**  
   - 신용카드 **부정 사용**, 보험 사기, 이상 거래 탐지 등에서 높은 부가가치를 창출.  
   - 과거 사기로 라벨링된 데이터를 활용해 준지도 학습을 수행, 정상 패턴에서 크게 어긋나는 거래를 실시간 차단[^8].

2. **보안(Security)**  
   - 침입 탐지 시스템(IDS)에서 대량의 트래픽·로그를 모니터링해, **비정상적 패킷 흐름**이나 **의심 로그인**을 포착[^8].  
   - 딥러닝은 과거에 없는 새로운 공격(제로데이 공격) 탐지에도 활용 가능.

3. **제조(Manufacturing)**  
   - **예지 정비(Predictive Maintenance)**: 공장 센서 데이터를 분석해 기계 이상 조기 경보.  
   - **영상 품질 검사**: 제품 표면 결함 자동 탐지, 불량품 선별. MVTec AD 등 공개 데이터셋이 대표적 사례[^9].  

4. **헬스케어(Healthcare)**  
   - 생체 신호 모니터링(심박수, 혈압 등)에서 급격한 이상 변화를 조기에 포착.  
   - **의료영상**(CT, MRI) 분석 시 정상 조직과 다른 병변 후보를 딥러닝으로 탐지, 진단 보조[^8].

5. **기타 응용**  
   - 자율주행차 센서 이상 탐지, 스마트시티 소음 감지, 클라우드 서버 로그 분석(AIOps) 등 산업 전반에서 활용되고 있음[^3].

---

## 4. 데이터셋 및 평가

1. **대표 데이터셋**  
   - **KDD Cup 1999**: 네트워크 침입 탐지의 고전적 벤치마크[^7].  
   - **NSL-KDD**, **Kyoto 2006+**: KDD Cup 99를 개선한 버전들.  
   - **MVTec AD**: 제조 이미지(부품, 섬유 등)에서 정상/불량 데이터를 대규모로 제공[^9].  
   - 기타 **신용카드 사기(IEEE-CIS)**, NASA 위성 센서(SMAP/MSL), Yahoo Webscope(웹 트래픽) 등 도메인별 데이터셋 존재.

2. **평가 지표**  
   - **ROC-AUC**: 이상치 비율이 달라져도 비교적 일관성 있게 모델을 평가.  
   - **정밀도(Precision)**, **재현율(Recall)**, **F1-스코어**, **PR-AUC** 등 임계값 기반 성능을 종합적으로 확인.  
   - 실제 운영환경에서는 거짓 경보(false alarm)와 놓침(missed detection) 간의 **Trade-off** 고려가 필수[^10].

---

## 5. 실무 적용과 한계

1. **적용 사례**  
   - 금융사: 실시간 거래 모니터링으로 부정 결제를 조기 차단.  
   - 클라우드 운영(AIOps): 서버 로그 이상 탐지로 장애 자동 복구.  
   - 제조: 센서 데이터·제품 이미지를 24시간 분석해 불량률 감소.  
   - 의료: MRI 등 의료영상 이상 부위를 자동 강조하여 조기 진단 보조[^8].

2. **한계점**  
   - **레이블 부족**: 이상치가 희소하고 형태가 다양해 지도학습이 어려움.  
   - **오탐·미탐** 문제: 경보를 과도하게 내거나 놓치는 사례가 모두 치명적.  
   - **블랙박스**: 딥러닝 모델의 **설명 가능성**이 낮아 금융·의료 등 엄격한 규제 분야에서 도입 장벽 존재.  
   - **환경 변화**: 데이터 분포가 변하는 **Concept Drift**나 새로운 유형의 이상에 대한 대응 필요.  
   - **실시간 처리 비용**: 대용량 스트림 데이터를 딥러닝 모델로 빠르게 처리하는 데 계산 자원 소모가 큼[^5].

이러한 문제들을 해결하기 위해서는 **사전학습**·**전이학습** 같은 기법, **모델 설명성**(XAI) 연구, **주기적 온라인 학습**, **전문가 피드백**을 결합한 하이브리드 운영 등 다양한 접근이 시도되고 있습니다[^5]. 현실 세계의 복잡한 이상 상황을 놓치지 않으면서도 운영 적합성을 갖추는 것이, **딥러닝 기반 이상 탐지**가 앞으로 해결해 나가야 할 중요한 과제입니다.

---

# 참고문헌

[^1]: Chalapathy, R., & Chawla, S. (2019). *Deep Learning for Anomaly Detection: A Survey*. **arXiv preprint arXiv:1901.03407**.  
[^2]: Ruff, L., Kauffmann, J. R., Vandermeulen, R. A., et al. (2022). *Self-Supervised Anomaly Detection: A Survey and Outlook*. **arXiv preprint arXiv:2205.05173**.  
[^3]: Yuan, X., Gong, Z., Tang, Y., et al. (2023). *A Survey of Deep Anomaly Detection in Multivariate Time Series: Taxonomy, Applications, and Directions*. **Sensors**, 25(1), 190.  
[^4]: Xu, H., Shen, W., Zheng, T., et al. (2021). *Time Series Anomaly Detection with Association Discrepancy*. **ICLR**.  
[^5]: Using Machine Learning for Time Series Anomaly Detection. (2022). [Online resource](https://www.anomalo.com/blog/machine-learning-approaches-to-time-series-anomaly-detection/)  
[^6]: *GANomaly — Anomalib documentation*. [Online resource](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/ganomaly.html)  
[^7]: Tavallaee, M., Bagchi, A., Lu, W., & Ghorbani, A. A. (2009). *A detailed analysis of the KDD CUP 99 data set*. **IEEE Symposium on Computational Intelligence for Security and Defense Applications**.  
[^8]: IBM. *Anomaly Detection in Machine Learning: Examples, Applications & Use Cases*. [Online resource](https://www.ibm.com/think/topics/machine-learning-for-anomaly-detection)  
[^9]: Bergmann, P., Batzner, K., et al. (2021). *The MVTec Anomaly Detection Dataset: A Survey on Unsupervised Industrial Image AD*. **arXiv preprint arXiv:2204.11161**.  
[^10]: Smith, J., Li, P., Nguyen, T., & Zhao, L. (2024). *Exploring the Impact of Outlier Variability on Anomaly Detection Evaluation Metrics*. **arXiv preprint arXiv:2409.15986**.

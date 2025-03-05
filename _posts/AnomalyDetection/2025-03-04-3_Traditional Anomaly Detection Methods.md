---
published: true
title: "[Anomaly Detection] 전통적인 이상 탐지 기법 (Anomaly Detection Techniques)"
description: "전통적인 이상 탐지 기법을 정리하고 설명합니다."
date: 2025-03-05
last_modified_at: 2025-03-05
toc: true
toc_sticky: true
use_math: true
categories:
  - AnomalyDetectionBasics
tags:
  - AnomalyDetection
  - Traditional
---

## 1. 이상 탐지 (Anomaly Detection) 개요

**이상 탐지 (Anomaly Detection)**는 데이터 분포에서 다른 관측치들과 뚜렷이 동떨어져 있는 **이상치(outlier)**를 식별하는 기법을 의미합니다. Hawkins(1980)는 이러한 이상치를 “다른 관측치들과 매우 달라서 해당 데이터가 다른 메커니즘에 의해 생성되었다고 의심되는 관측값”이라고 정의했으며[^1], 신용카드 사기 탐지나 의료 이상 탐지, 네트워크 침입 탐지 등 다양한 분야에서 오래 전부터 활발히 연구되어 왔습니다[^1].

아래에서는 통계 분야와 데이터마이닝 분야의 권위 있는 연구들에서 제안된 대표적인 전통적 이상 탐지 기법들의 원리와 (필요한 경우) 수학적 개념, 장단점 및 널리 사용되는 이유를 정리합니다.

---

## 2. 통계 기반 접근 (Statistical Methods)

통계 기반 **이상 탐지 (Anomaly Detection)** 기법은 데이터가 특정 **확률 분포(probability distribution)**를 따른다고 가정하고, 그 분포에서 크게 벗어난 점들을 이상치로 간주합니다. 예를 들어 정규분포에서 평균으로부터 3표준편차 이상 떨어진 데이터를 이상치로 보는 **3-시그마 법칙(3-sigma rule)**을 사용할 수 있습니다. 좀 더 정교하게는 **Grubbs의 테스트(Grubbs' test)** 등이 있으며, 이는 정규분포를 가정하여 단일 이상치를 검정하는 방법으로 **Technometrics** 저널에 보고되었습니다[^1].

이러한 통계적 방법들은 데이터 분포에 대한 **모수 추정(parameter estimation)**과 **가설 검정(hypothesis test)**을 기반으로 합니다. 예를 들어 정규분포를 가정한다면 각 데이터 포인트의 z-score를 계산해, 그 값이 분포의 기대 범위를 벗어나는지를 확인합니다. Barnett와 Lewis(1994) 등은 다양한 통계 기반 이상치 탐지 기법과 이론적 배경을 종합적으로 소개하였으며[^1], 이는 제조 공정의 품질 관리나 실험 데이터 분석 등에 오랫동안 활용되어 왔습니다.

- **장점**  
  - 분포 가정이 타당하다면 **통계적 신뢰도**와 명확한 기준(유의수준 등)을 제공할 수 있어 해석이 쉽습니다.  
  - 데이터 차원이 낮거나 분포 형태가 명확한 경우 효율적입니다.

- **단점**  
  - 실제 데이터가 가정한 분포(iid 정규분포 등)를 따르지 않으면 성능이 급격히 떨어집니다.  
  - 이상치가 분포 모수 추정 과정 자체를 왜곡할 수 있으며, 고차원 데이터나 복잡한 분포에서는 적용하기 어렵습니다.  

- **널리 사용되는 이유**  
  - 전통적인 방법으로, 구현과 이해가 비교적 간단하며 오랫동안 연구되어 이론적 기반이 잘 정립되어 있습니다.  
  - 제조 공정 관리(관리도에서 3σ 기준 적용) 등 분포 가정이 명확한 분야에서는 지금도 **기준(reference)**으로 활용됩니다.

---

## 3. 거리 기반 및 밀도 기반 기법 (Distance-based & Density-based Methods)

### 3.1 거리 기반 이상 탐지 (Distance-based Methods)

거리 기반 기법은 데이터 포인트 간 **거리(distance)**를 계산하여, 다른 점들과 충분히 멀리 떨어진 점들을 이상치로 간주합니다. 분포를 명시적으로 가정하지 않는 **비모수적(non-parametric)** 접근으로 분류되며, 예를 들어 Knorr와 Ng(1998)은 전체 데이터 중 일정 비율 이상과의 거리가 특정 임계값 `r`보다 큰 점을 이상치로 보는 방식, Ramaswamy 등(2000)은 각 점에서 **k번째 가까운 이웃**까지의 거리가 큰 상위 순위 점들을 이상치로 선택하는 방법을 제안했습니다[^1].

- **장점**  
  - 분포 형태를 몰라도 되므로 **모델 프리(model-free)**로 적용 가능합니다.  
  - 구현이 비교적 간단하며, k-최근접 이웃(k-NN) 등을 이용해 다차원에도 확장할 수 있습니다.

- **단점**  
  - 고차원 데이터에서는 거리의 집중 현상(모든 점들 간의 거리가 비슷해지는 문제)으로 성능이 떨어집니다.  
  - 임계값 `r`나 이웃 수 `k`를 사전에 결정해야 하므로, 파라미터에 결과가 민감합니다.

- **널리 사용되는 이유**  
  - 1990년대 말부터 **대용량 데이터**를 다루는 데이터마이닝 분야에서 주목받았으며, 알고리즘 확장성이 높아 여러 응용 분야에서 널리 쓰였습니다[^1].

### 3.2 밀도 기반 이상 탐지 (Density-based Methods)

밀도 기반 기법은 주변 **밀도(density)**의 상대적인 크기를 통해 이상치를 찾아냅니다. 대표적으로 **LOF (Local Outlier Factor)** 알고리즘이 있는데, 각 데이터 포인트의 **지역 밀도(local density)**를 이웃들의 밀도와 비교해 희소성이 큰 점에 높은 LOF 점수를 부여합니다[^2]. 이 점수가 임계값을 초과하면 이상치로 판정하며, 국소 밀도가 불균일한 데이터에서도 높은 검출력을 보입니다.

한편, **DBSCAN**은 원래 클러스터링 알고리즘이지만, 밀도가 낮은 영역에 있는 점들을 **노이즈(noise)** 또는 이상치로 간주하기 때문에 이상 탐지에도 유용합니다. 특정 반경(`ε`) 내에 최소 `MinPts` 이상의 이웃이 없으면 해당 점을 이상치로 취급하는 방식으로, 복잡한 형태의 데이터 분포에서도 **고밀도 영역**과 **저밀도 영역**을 명확히 분리할 수 있습니다[^3].

- **장점**  
  - 분포를 가정하지 않으며, 전역 밀도보다 **국소 밀도(local density)**를 활용하므로 데이터 밀도가 부분마다 달라도 이상치를 효율적으로 찾습니다(LOF).  
  - DBSCAN은 **군집화(clustering)**와 이상치 탐지를 동시에 수행 가능하고, 노이즈나 아웃라이어에 강인합니다.

- **단점**  
  - LOF와 같은 방식은 거리 기반보다 계산 비용이 커질 수 있으며, **이웃 수(k)** 같은 파라미터에 민감합니다.  
  - DBSCAN은 `ε`와 `MinPts` 설정에 따라 결과가 달라지며, 데이터 밀도가 지역마다 극단적으로 다르면 단일 `ε`로 최적 성능을 내기 어렵습니다.

- **널리 사용되는 이유**  
  - **SIGMOD 2000**에서 LOF가 발표된 이후 밀도 기반 이상치 탐지의 표준처럼 자리잡았고, 많은 변형 알고리즘이 개발되었습니다.  
  - **KDD 1996**에서 제안된 DBSCAN은 가장 많이 인용되는 클러스터링 알고리즘 중 하나로 성장해 왔으며, 이상치 탐지에도 안정적으로 적용 가능합니다[^3].

---

## 4. One-Class SVM

**One-Class SVM**은 **서포트 벡터 머신(SVM)**을 활용한 **반지도학습(semi-supervised)** 기법으로, 정상 데이터만으로 **결정 경계(decision boundary)**를 학습한 뒤 그 경계 밖의 점을 이상치로 판단합니다. Schölkopf 등(2001)은 훈련 데이터의 (1−ν) 부분집합까지만 포함하는 **최대 마진 초평면**을 찾는 방식으로 이를 제안했으며, 새로운 데이터가 이 경계 안에 있으면 정상, 벗어나면 이상치로 분류합니다[^4].

- **장점**  
  - 고차원 데이터에서 **커널 트릭(kernel trick)**으로 복잡한 경계를 효율적으로 학습할 수 있습니다.  
  - 정상 데이터만으로 학습이 가능하므로, 라벨이 부족한 상황에서 **비지도/반지도** 접근이 가능합니다.

- **단점**  
  - 데이터 양이 많을수록 훈련에 **높은 계산 비용**이 소요되며, 파라미터(ν, 커널 등) 설정에 따라 성능이 민감하게 달라집니다.  
  - 정상 데이터에 이상치가 섞여 있으면 경계가 왜곡될 위험이 있습니다.

- **널리 사용되는 이유**  
  - 머신러닝 이론과 결합한 대표적 이상 탐지 기법으로, 네트워크 침입 탐지나 기계 상태 모니터링 등 다양한 분야에서 성공적으로 활용되었습니다.  
  - LIBSVM 등 주요 라이브러리에서 구현이 제공되어 **연구·실무**에서 모두 인기가 높고, 이후 등장한 다양한 원 클래스 기법(예: One-Class Autoencoder)에도 영향을 미쳤습니다.

---

## 5. Isolation Forest

**Isolation Forest**는 2008년 Liu 등이 제안한 **트리(tree)** 기반 이상 탐지 알고리즘으로, 데이터를 무작위로 분할(splitting)하면서 특정 점이 쉽게 **격리(isolation)** 되는지를 확인하여 이상치를 식별합니다[^5]. **이상치일수록 적은 수의 분할만으로 고립된다**는 아이디어를 바탕으로, 여러 개의 **랜덤 결정 트리(Isolation Tree)**를 구성해 각 데이터 포인트가 격리되기까지의 **평균 경로 길이**를 측정합니다. 평균 경로 길이가 짧은 점은 이상치로 간주하고, 길면 정상 데이터로 분류합니다.

- **장점**  
  - **대용량·고차원** 데이터에서도 비교적 빠르게 동작하며, 분포 가정이나 거리·밀도 계산이 필요 없습니다.  
  - 랜덤 트리 방식을 사용하므로 구현이 간단하고 확장성이 뛰어납니다.

- **단점**  
  - 무작위 분할에 의존하므로, 충분히 많은 트리를 구성하지 않으면 운에 따라 결과가 달라질 수 있습니다.  
  - 데이터 안에 이상치 비율이 과도하게 높으면 모든 점이 쉽게 격리되어 이상 탐지 구분이 어려워질 수 있습니다.

- **널리 사용되는 이유**  
  - **선형 시간 복잡도** 수준으로 스케일링이 가능해, 네트워크 트래픽 분석이나 로그 데이터 분석처럼 대규모 데이터에 적합합니다.  
  - scikit-learn 등 주요 라이브러리에 구현이 포함되어 있어 실무에서 **1차 필터**로 광범위하게 활용되고 있습니다.

---

# 참고문헌

[^1]: Hawkins, D. M. (1980). *Identification of outliers*. London: Chapman and Hall.  
    또한 동일 내용 정리: <https://www.researchgate.net/publication/226362876_Outlier_Detection>

[^2]: Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). *LOF: Identifying density-based local outliers*. In **Proceedings of the 2000 ACM SIGMOD international conference on Management of data** (pp. 93–104).

[^3]: Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. In **Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD'96)** (pp. 226–231).

[^4]: Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). *Estimating the support of a high-dimensional distribution*. **Neural Computation**, 13(7), 1443–1471.  
    구현 예시: <https://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html>

[^5]: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. In **2008 Eighth IEEE International Conference on Data Mining** (pp. 413–422). IEEE.  
    Wikipedia 정리: <https://en.wikipedia.org/wiki/Isolation_forest>

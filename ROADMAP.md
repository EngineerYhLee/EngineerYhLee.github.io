# Research Roadmap for Medical Image Translation

## 1. Programming

### 1.1 파이썬 (Python)
- [ ] **파이썬 설치 및 환경 설정 (Python Installation & Environment Setup)**
  - [ ] 파이썬 설치 (Install Python)
  - [ ] 파이썬 버전 관리 (Python Version Management)
  - [ ] PIP 사용법 (Using PIP)
  - [ ] 가상 환경 설정 (Virtual Environment Setup)

- [ ] **파이썬 기본 문법 (Python Basic Syntax)**
  - [ ] 변수 (Variables)
  - [ ] 데이터 타입 (Data Types)
  - [ ] 연산자 (Operators)
  - [ ] 조건문 (Conditional Statements)
  - [ ] 반복문 (Loops)
  - [ ] 함수 (Functions)
  - [ ] 클래스 (Classes)
  - [ ] 예외 처리 (Exception Handling)
  - [ ] 리스트 (Lists)
  - [ ] 딕셔너리 (Dictionaries)
  - [ ] 제너레이터 (Generators)
  - [ ] 이터레이터 (Iterators)
  - [ ] 데코레이터 (Decorators)
  - [ ] 컨텍스트 매니저 (Context Managers)

- [ ] **프로젝트 구조화 (Project Structuring)**
  - [ ] 모듈화 및 코드 재사용 (Modularization & Code Reusability)
  - [ ] 프로젝트 디렉토리 구조 (Project Directory Structure)
  - [ ] 버전 관리 (Git)
  - [ ] 패키지 관리 (예: Poetry)

- [ ] **파이썬 라이브러리 활용 (Using Python Libraries)**
  - [ ] NumPy
  - [ ] Pandas
  - [ ] Matplotlib

---

### 1.2 파이토치 (PyTorch)
- [ ] **환경 설정 (Environment Setup)**
  - [ ] 파이토치 설치 (Install PyTorch)
  - [ ] 쿠다(CUDA) 설치 (Install CUDA)
  - [ ] 버전 호환성 확인 (Check Version Compatibility)

- [ ] **텐서 기초 (Tensor Basics)**
  - [ ] 텐서 개념과 생성 (Tensor Concept & Creation)
  - [ ] 텐서 데이터 타입 (Tensor Data Types)
  - [ ] 텐서 연산 (산술, 선형대수) (Tensor Operations: Arithmetic & Linear Algebra)
  - [ ] 텐서 인덱싱과 슬라이싱 (Tensor Indexing & Slicing)
  - [ ] 텐서 형상 조작 (reshape, view) (Tensor Reshaping)
  - [ ] 디바이스 간 이동 (CPU/GPU) (Moving Tensors Between Devices)

- [ ] **자동 미분 (Autograd)**
  - [ ] 계산 그래프 이해 (Understanding Computational Graphs)
  - [ ] requires_grad 설정 (Setting requires_grad)
  - [ ] backward() 함수 (Using backward())
  - [ ] 그래디언트 계산과 초기화 (Gradient Calculation & Zeroing)
  - [ ] 사용자 정의 autograd 함수 (Custom Autograd Functions)

- [ ] **신경망 모듈 (nn.Module)**
  - [ ] nn.Module 상속과 활용 (Subclassing nn.Module)
  - [ ] 기본 레이어 (Linear, Conv2d 등) (Basic Layers)
  - [ ] 활성화 함수 (Activation Functions)
  - [ ] 손실 함수 (Loss Functions)
  - [ ] 모델 파라미터 관리 (Model Parameter Management)
  - [ ] forward() 구현 (Implementing forward())
  - [ ] Sequential을 활용한 모델 구성 (Using nn.Sequential)

- [ ] **최적화 (Optimization)**
  - [ ] 옵티마이저 (Optimizers)
  - [ ] 스케줄러 (Learning Rate Scheduler)
  - [ ] 그래디언트 클리핑 (Gradient Clipping)
  - [ ] 가중치 초기화 기법 (Weight Initialization)

- [ ] **데이터 로딩과 처리 (Data Loading & Processing)**
  - [ ] Dataset 클래스 구현 (Implementing Dataset)
  - [ ] DataLoader 사용 (Using DataLoader)
  - [ ] 배치 처리 (Batch Processing)
  - [ ] 데이터 전처리와 변환 (Data Preprocessing & Transformations)
  - [ ] 커스텀 데이터셋 생성 (Creating Custom Datasets)

- [ ] **모델 저장과 불러오기 (Saving & Loading Models)**
  - [ ] state_dict 이해 (Understanding state_dict)
  - [ ] 체크포인트 저장 (Checkpointing)
  - [ ] 모델 로드 (Loading Saved Models)
  - [ ] JIT와 TorchScript (JIT & TorchScript)

- [ ] **분산 학습 (Distributed Training)**
  - [ ] DistributedDataParallel 사용 (Using DistributedDataParallel)
  - [ ] 멀티 GPU 학습 (Multi-GPU Training)
  - [ ] 분산 학습 설정 (Distributed Setup)

- [ ] **성능 최적화 (Performance Optimization)**
  - [ ] 프로파일링 도구 사용 (Using Profiling Tools)
  - [ ] 메모리 최적화 (Memory Optimization)
  - [ ] 연산 최적화 (Operation Optimization)
  - [ ] torch.compile 사용 (Using torch.compile)
  - [ ] 양자화 (Quantization)
  - [ ] 모델 가지치기 (Pruning)

- [ ] **확장 라이브러리 (Libraries)**
  - [ ] **TorchVision:** 내장 및 커스텀 이미지 데이터셋, 이미지 전처리, 데이터 증강, 사전학습 모델, 전이 학습, 이미지 I/O, 시각화
  - [ ] **TorchAudio:** 내장 및 커스텀 오디오 데이터셋, 파형 처리, 스펙트로그램, MFCC 추출
  - [ ] **TorchText:** 토큰화, 정규화, 어휘 사전 구축, 내장 텍스트 데이터셋, 시퀀스 모델링

---

## 2. Mathematics

### 2.1 함수 (Functions)
- [ ] **함수의 정의 (Definition of Functions)**
  - [ ] 정의역, 치역, 공역 (Domain, Range, Codomain)
  - [ ] 다항함수, 지수함수, 로그함수, 삼각함수 (Polynomial, Exponential, Logarithmic, Trigonometric Functions)
  - [ ] 연속성과 극한 (Continuity & Limits)

- [ ] **그래프 해석 및 도구 활용 (Graph Interpretation & Tools)**
  - [ ] 평행 이동, 스케일링 (Translations & Scalings)
  - [ ] 그래프 시각화 도구 (Plotting Libraries)

- [ ] **합성 함수와 역함수 (Composite & Inverse Functions)**
  - [ ] 합성 함수 (Composite Functions: f(g(x)))
  - [ ] 역함수 정의 및 계산 (Inverse Functions)

---

### 2.2 선형대수 (Linear Algebra)
- [ ] **벡터와 벡터 공간 (Vectors & Vector Spaces)**
  - [ ] 벡터 연산 (덧셈, 스칼라 곱) (Vector Operations)
  - [ ] 부분공간 (Subspaces)
  - [ ] 내적, 외적, 노름 (Inner Product, Outer Product, Norm)

- [ ] **행렬과 행렬 연산 (Matrices & Matrix Operations)**
  - [ ] 행렬 덧셈, 곱셈, 전치 (Addition, Multiplication, Transpose)
  - [ ] 행렬식과 역행렬 (Determinant & Inverse)
  - [ ] 랭크 (Rank)

- [ ] **선형 독립성, 기저, 차원 (Linear Independence, Basis, Dimension)**
  - [ ] 선형 독립성과 종속성 (Linear Independence vs. Dependence)
  - [ ] 기저와 차원 (Basis & Dimension)

- [ ] **고유값, 고유벡터, 분해 기법 (Eigenvalues, Eigenvectors & Decomposition)**
  - [ ] 고유값, 고유벡터 (Eigenvalues & Eigenvectors)
  - [ ] 특이값 분해(SVD), LU, QR, Cholesky (SVD, LU, QR, Cholesky)

- [ ] **직교 변환과 대각화 (Orthogonal Transformations & Diagonalization)**
  - [ ] 직교 행렬 (Orthogonal Matrices)
  - [ ] 대각화와 정사영 (Diagonalization & Projections)

---

### 2.3 미적분 (Calculus)
- [ ] **미분 (Differentiation)**
  - [ ] 도함수와 룰 (Derivatives & Rules)
  - [ ] 편미분, 다변수 미분 (Partial Derivatives, Multivariable Calculus)
  - [ ] 기울기(Gradient), 야코비안(Jacobian), 헤시안(Hessian)

- [ ] **적분 (Integration)**
  - [ ] 부정적분, 정적분 (Indefinite & Definite Integrals)
  - [ ] 적분 기법 (치환, 부분적분 등) (Integration Techniques)
  - [ ] 이중·삼중적분 (Double & Triple Integrals)

- [ ] **미분방정식 (Differential Equations)**
  - [ ] 1차 미분방정식 (1st Order ODEs)
  - [ ] 고계 미분방정식 (Higher-Order ODEs)
  - [ ] 해의 존재성 (Existence & Uniqueness)

- [ ] **라플라스 변환 (Laplace Transform)**
  - [ ] 라플라스 변환 정의와 성질 (Definition & Properties)
  - [ ] 미분방정식 풀이 (Solving ODEs)

- [ ] **푸리에 변환 (Fourier Transform)**
  - [ ] 푸리에 급수, 푸리에 변환 (Fourier Series & Transform)
  - [ ] 신호·이미지 처리 응용 (Signal & Image Processing)

---

### 2.4 확률과 통계 (Probability & Statistics)
- [ ] **확률 (Probability)**
  - [ ] 확률의 공리 (Axioms of Probability)
  - [ ] 조건부 확률과 독립성 (Conditional Probability & Independence)
  - [ ] 베이즈 정리 (Bayes’ Theorem)

- [ ] **확률변수와 분포 (Random Variables & Distributions)**
  - [ ] 이산/연속 확률변수 (Discrete & Continuous Random Variables)
  - [ ] 주요 분포 (이항, 포아송, 정규 등) (Key Distributions)
  - [ ] PMF, PDF, CDF

- [ ] **기대값과 분산 (Mean & Variance)**
  - [ ] 기대값, 분산, 표준편차 (Mean, Variance, Standard Deviation)
  - [ ] 고차 모멘트 (Higher-Order Moments)

- [ ] **추정 이론과 통계적 개념 (Estimation & Statistical Concepts)**
  - [ ] 최대우도추정(MLE) (Maximum Likelihood Estimation)
  - [ ] 공분산, 상관계수 (Covariance & Correlation)
  - [ ] 점추정, 구간추정, 가설검정 (Point Estimation, Interval Estimation, Hypothesis Testing)

- [ ] **법칙들 (Laws)**
  - [ ] 대수의 법칙 (Law of Large Numbers)
  - [ ] 중심극한정리 (Central Limit Theorem)

---

## 3. Computer Vision

### 3.1 이미지 처리 기초 (Image Processing Basics)
- [ ] **이미지 기본 구조 (Basic Image Structure)**
  - [ ] 픽셀, 채널, 해상도 (Pixels, Channels, Resolution)
  - [ ] 이미지 크기 조정 (Resizing)
  - [ ] 컬러 스페이스 (RGB, HSV, Lab 등) (Color Spaces)

- [ ] **기본 필터링 및 연산 (Basic Filtering & Operations)**
  - [ ] 저주파/고주파 필터 (Gaussian, Sobel 등) (Low/High Pass Filters)
  - [ ] 선형/비선형 필터 (Median, Bilateral 등) (Linear/Non-linear Filters)
  - [ ] 모폴로지 연산 (침식, 팽창 등) (Morphological Operations)

- [ ] **주파수 도메인 처리 (Frequency Domain Processing)**
  - [ ] 푸리에 변환(DFT), DCT, 웨이블릿 변환 (Fourier, DCT, Wavelet)
  - [ ] 주파수 영역에서의 이미지 향상 (Frequency-Domain Image Enhancement)

- [ ] **기하학적 변환 (Geometric Transformations)**
  - [ ] 회전, 스케일 조정, 반전 (Rotation, Scaling, Flipping)
  - [ ] 투영 변환, 시점 변환 (Perspective Transformation)

- [ ] **이미지 히스토그램 (Image Histograms)**
  - [ ] 히스토그램 계산, 평활화 (Histogram Calculation & Equalization)

---

### 3.2 특징 추출 (Feature Extraction)
- [ ] **코너 검출 (Corner Detection)**
  - [ ] 해리스 코너 (Harris), 쉬-토마시(Shi-Tomasi) (Harris, Shi-Tomasi)
  - [ ] FAST, BRIEF 등 (FAST, BRIEF)

- [ ] **로컬 특징점 검출 및 서술자 (Local Feature Detectors & Descriptors)**
  - [ ] SIFT, SURF, ORB (SIFT, SURF, ORB)
  - [ ] KAZE, AKAZE (KAZE, AKAZE)
  - [ ] 키포인트 매칭과 RANSAC (Keypoint Matching & RANSAC)

- [ ] **HOG (Histogram of Oriented Gradients)**
  - [ ] HOG 특징 추출 (HOG Feature Extraction)
  - [ ] 객체 검출 응용 (HOG + SVM 등) (Object Detection)

- [ ] **텍스처 분석 (Texture Analysis)**
  - [ ] 가보(Gabor) 필터 (Gabor Filters)
  - [ ] LBP (Local Binary Patterns)

---

### 3.3 객체 검출 및 추적 (Object Detection & Tracking)
- [ ] **전통적 객체 검출 기법 (Traditional Object Detection)**
  - [ ] 하르 캐스케이드 (Haar Cascades)
  - [ ] HOG+SVM (HOG+SVM)
  - [ ] 배경 차분 (Background Subtraction)

- [ ] **이미지 분할 기법 (Image Segmentation Basics)**
  - [ ] 임계값 기반 (Thresholding, Otsu 등) (Thresholding)
  - [ ] 엣지 검출 (Sobel, Canny 등) (Edge Detection)

- [ ] **객체 추적 (Object Tracking)**
  - [ ] KCF, GOTURN 등 (KCF, GOTURN)
  - [ ] SORT, DeepSORT 등 멀티 객체 추적 (Multi-Object Tracking)
  - [ ] 옵티컬 플로우 (Optical Flow; Lucas-Kanade, Farneback)

---

### 3.4 3D 비전 (3D Vision)
- [ ] **3D 이미지 처리 기초 (3D Image Processing Basics)**
  - [ ] 깊이 영상, RGB-D 데이터 (Depth Maps, RGB-D)
  - [ ] 스테레오 비전 (Stereo Vision)
  - [ ] 다중 뷰 기하학 (Multi-View Geometry)

- [ ] **포인트 클라우드 (Point Cloud)**
  - [ ] 구조, 시각화 (Structure & Visualization)
  - [ ] 정합(Registration), ICP (Registration & ICP Algorithm)
  - [ ] 필터링, 노이즈 제거 (Filtering & Denoising)

- [ ] **3D 객체 인식 및 분할 (3D Object Recognition & Segmentation)**
  - [ ] PointNet, PointNet++ 등 (PointNet, PointNet++)
  - [ ] VoxelNet, MinkowskiNet (3D Segmentation Models)

- [ ] **3D 재구성 및 SLAM (3D Reconstruction & SLAM)**
  - [ ] 구조에서 모션(SfM) (Structure from Motion)
  - [ ] SLAM (ORB-SLAM, LSD-SLAM 등)
  - [ ] 메쉬 생성, 텍스처 매핑 (Mesh & Texture Mapping)

---

## 4. Machine Learning

### 4.1 지도 학습 (Supervised Learning)
- [ ] **회귀 (Regression)**
  - [ ] 선형 회귀 (Linear Regression)
  - [ ] 다항 회귀 (Polynomial Regression)
  - [ ] 릿지, 라쏘, 엘라스틱넷 (Ridge, Lasso, Elastic Net)
  - [ ] 회귀 평가 지표 (MSE, MAE, R² 등) (Regression Metrics)

- [ ] **분류 (Classification)**
  - [ ] 로지스틱 회귀 (Logistic Regression)
  - [ ] 결정 트리 (Decision Trees)
  - [ ] 서포트 벡터 머신 (SVM)
  - [ ] k-최근접 이웃 (k-NN)
  - [ ] 나이브 베이즈 (Naive Bayes)
  - [ ] 분류 평가 지표 (정밀도, 재현율, F1 등) (Classification Metrics)

- [ ] **앙상블 (Ensemble)**
  - [ ] 배깅/랜덤 포레스트 (Bagging/Random Forest)
  - [ ] 부스팅 (AdaBoost, XGBoost, LightGBM 등) (Boosting)
  - [ ] 스태킹 (Stacking)

- [ ] **특성 공학 및 정규화 (Feature Engineering & Regularization)**
  - [ ] 스케일링 (표준화, 정규화) (Scaling)
  - [ ] 원-핫 인코딩 (One-Hot Encoding)
  - [ ] PCA, LDA (차원 축소) (Dimensionality Reduction)
  - [ ] 과적합 방지 기법 (Overfitting Prevention)

---

### 4.2 비지도 학습 (Unsupervised Learning)
- [ ] **군집화 (Clustering)**
  - [ ] k-평균 (k-Means)
  - [ ] 계층적 클러스터링 (Hierarchical Clustering)
  - [ ] DBSCAN, OPTICS 등 (Density-Based Methods)
  - [ ] 가우시안 혼합 모델 (GMM) (Gaussian Mixture Model)

- [ ] **차원 축소 (Dimensionality Reduction)**
  - [ ] PCA (Principal Component Analysis)
  - [ ] t-SNE, UMAP (t-SNE, UMAP)
  - [ ] LDA (Linear Discriminant Analysis)
  - [ ] 오토인코더 기반 (Autoencoder-Based, 선택)

- [ ] **특징 학습 (Feature Learning)**
  - [ ] 자기지도 (Self-Supervised), 표현 학습 (Representation Learning)
  - [ ] 잠재 변수 모델 (VAE, GAN 기초) (Latent Variable Models)

---

### 4.3 모델 평가 (Model Evaluation)
- [ ] **데이터 분할 및 검증 (Data Splitting & Validation)**
  - [ ] 홀드아웃 (Train/Val/Test) (Hold-out)
  - [ ] 교차 검증 (K-Fold, Cross-Validation)

- [ ] **분류 평가 지표 (Classification Metrics)**
  - [ ] 혼동 행렬 (Confusion Matrix)
  - [ ] 정밀도, 재현율, F1 (Precision, Recall, F1)
  - [ ] ROC 곡선, AUC, PR 곡선 (ROC, AUC, PR Curve)

- [ ] **회귀 평가 지표 (Regression Metrics)**
  - [ ] MSE, MAE, RMSE
  - [ ] R² (결정계수, Coefficient of Determination)

- [ ] **불균형 데이터셋 (Imbalanced Dataset)**
  - [ ] 오버샘플링(SMOTE), 언더샘플링 (Oversampling, Undersampling)
  - [ ] Focal Loss 등 활용 (Focal Loss)
  - [ ] ROC AUC, PR Curve 강조 (AUC, PR Emphasis)

- [ ] **모델 선택 및 성능 비교 (Model Selection & Comparison)**
  - [ ] 그리드 서치 (Grid Search), 랜덤 서치 (Random Search)
  - [ ] 베이지안 최적화 (Bayesian Optimization)

---

### 4.4 최적화 (Optimization)
- [ ] **경사 하강법 기초 (Gradient Descent Basics)**
  - [ ] 배치, 미니배치, 확률적 경사 하강법 (Batch, Mini-Batch, Stochastic GD)
  - [ ] 학습률 스케줄링 (Learning Rate Scheduling)

- [ ] **확률적 경사 하강법 (Stochastic Gradient Descent)**
  - [ ] SGD 알고리즘 이해 (Understanding SGD)
  - [ ] 모멘텀, 네스테로프 가속경사 (Momentum, Nesterov)

- [ ] **적응형 최적화 알고리즘 (Adaptive Optimizers)**
  - [ ] RMSprop
  - [ ] Adam, AdamW
  - [ ] Adagrad, Adadelta

- [ ] **하이퍼파라미터 튜닝 (Hyperparameter Tuning)**
  - [ ] Grid Search, Random Search
  - [ ] Hyperopt, Optuna 등 (Bayesian Optimization Tools)
  - [ ] 주요 초매개변수 최적화 (Learning Rate, Regularization 등)

---

## 5. Deep Learning

### 5.1 신경망 기초 (Neural Network Basics)
- [ ] **퍼셉트론, 다층 퍼셉트론 (Perceptron & MLP)**
  - [ ] 퍼셉트론 이론 (Perceptron Theory)
  - [ ] MLP 구조와 작동 원리 (Architecture & Mechanisms)
  - [ ] 초기화, 활성화 함수 중요성 (Initialization, Activation)

- [ ] **활성화 함수 (Activation Functions)**
  - [ ] ReLU, Sigmoid, Tanh
  - [ ] Leaky ReLU, ELU 등 (Variants)
  - [ ] 선택 기준 및 장단점 (Selection Criteria, Pros/Cons)

- [ ] **손실 함수 (Loss Functions)**
  - [ ] 크로스 엔트로피 (Cross-Entropy)
  - [ ] MSE, MAE, Huber 등 (Regression Losses)

- [ ] **역전파와 최적화 (Backpropagation & Optimization)**
  - [ ] 역전파 알고리즘 개념 (Backprop Algorithm)
  - [ ] 자동 미분 (Autograd) 활용 (Using Autograd)
  - [ ] 기본 옵티마이저 (GD, SGD, Momentum 등) (Basic Optimizers)

- [ ] **정규화와 일반화 (Regularization & Generalization)**
  - [ ] 드롭아웃 (Dropout), 배치 정규화 (Batch Normalization)
  - [ ] 조기 종료 (Early Stopping), 가중치 감쇠 (Weight Decay)

---

### 5.2 컨볼루션 신경망 (Convolutional Neural Networks, CNN)
- [ ] **컨볼루션과 풀링 (Convolution & Pooling)**
  - [X] CNN 개요 (CNN Summary)
  - [X] CNN 역사 (CNN History)
  - [ ] 컨볼루션 커널 (Convolution Kernels)
  - [ ] 패딩, 스트라이드 영향 (Padding & Stride)
  - [ ] 풀링 계층 (Pooling Layer)
  - [ ] 완전 연결 계층 (Fully Connected Layer)
  - [ ] 컨볼루션 레이어 쌓기 (Stacked Convolution Layers)

- [ ] **대표 CNN 아키텍처 (Representative CNN Architectures)**
  - [ ] LeNet
  - [ ] AlexNet
  - [ ] VGG
  - [ ] ResNet
  - [ ] Inception
  - [ ] DenseNet
  - [ ] MobileNet
  - [ ] EfficientNet

- [ ] **고급 CNN 기법 (Advanced CNN Techniques)**
  - [ ] 딜레이티드 컨볼루션 (Dilated Convolution)
  - [ ] 그룹 컨볼루션, 셔플넷 (Grouped Convolution, ShuffleNet)
  - [ ] 어텐션 결합 (Attention Integration)

---

### 5.3 순환 신경망 (Recurrent Neural Networks, RNN)
- [ ] **RNN 기본 구조 (RNN Basics)**
  - [ ] 순환 연결 (Recurrent Connections)
  - [ ] BPTT (Backpropagation Through Time)
  - [ ] 기울기 소실/폭주 문제 (Vanishing/Exploding Gradients)

- [ ] **LSTM, GRU (LSTM & GRU)**
  - [ ] 셀 상태, 게이트 구조 (Cell States, Gates)
  - [ ] GRU와의 차이점 (Difference from GRU)
  - [ ] 활용 사례 (Use Cases)

- [ ] **시퀀스-투-시퀀스 (Sequence-to-Sequence)**
  - [ ] 인코더-디코더 구조 (Encoder-Decoder)
  - [ ] 기계 번역, 챗봇 등 (Machine Translation, Chatbots)

- [ ] **어텐션 메커니즘 (Attention Mechanism)**
  - [ ] Bahdanau, Luong 어텐션 (Bahdanau, Luong Attention)
  - [ ] Dot-Product, Additive, Scaled Dot-Product
  - [ ] 다양한 시퀀스 처리 적용 (Various Sequence Tasks)

---

### 5.4 심화 주제 (Advanced Topics)
- [ ] **메모리 네트워크 (Memory Networks)**
  - [ ] Neural Turing Machine (NTM), DNC 등 (Differentiable Neural Computer)

- [ ] **강화 학습 기초 (Reinforcement Learning Basics)**
  - [ ] Q-러닝, DQN (Q-Learning, DQN)
  - [ ] 정책 경사 (Policy Gradient Concepts)

- [ ] **미세 조정과 도메인 적응 (Fine-Tuning & Domain Adaptation)**
  - [ ] 도메인 시프트 대응 (Domain Shift)
  - [ ] 소규모 데이터, 불균형 데이터 처리 (Handling Small/Imbalanced Data)

- [ ] **모델 경량화 (Model Compression)**
  - [ ] 양자화 (Quantization), 가지치기 (Pruning)
  - [ ] 지식 증류 (Knowledge Distillation)

---

## 6. Medical Imaging

### 6.1 의료 이미지 모달리티 (Medical Image Modalities)
- [ ] **X-ray, CT, MRI, 초음파 (X-ray, CT, MRI, Ultrasound)**
  - [ ] 물리학적 영상 획득 원리 (Physics of Acquisition)
  - [ ] CT: 흡수율 기반, MRI: T1/T2/FLAIR 시퀀스 (CT, MRI Sequences)
  - [ ] 초음파: 음파 반사, 도플러 영상 (Ultrasound, Doppler)

- [ ] **핵의학 영상 (PET, SPECT 등) (Nuclear Medicine)**
  - [ ] 방사성 동위원소 원리 (Radioisotope Principles)
  - [ ] PET, SPECT (PET, SPECT)
  - [ ] 기능 영상 vs 해부학적 영상 (Functional vs Anatomical)

- [ ] **의료 이미지 특수성 (Special Considerations)**
  - [ ] 해상도, 픽셀 스페이싱, 슬라이스 두께 (Resolution, Pixel Spacing, Slice Thickness)
  - [ ] 노이즈 종류 (전자적, 양자 등) (Noise Types)
  - [ ] 아티팩트 (금속, 모션 등) (Artifacts)
  - [ ] 의료영상 표준 (DICOM, NIfTI) (Medical Imaging Standards)

---

### 6.2 데이터 전처리 및 증강 (Image Preprocessing & Augmentation)
- [ ] **의료 이미지 포맷 및 처리 (Medical Image Formats & Processing)**
  - [ ] DICOM, NIfTI 등 (DICOM, NIfTI)
  - [ ] 윈도잉(CT Window), LUT 적용 (Windowing, LUT)
  - [ ] 공간 해상도, 방향성, 메타데이터 (Spatial Resolution, Orientation, Metadata)

- [ ] **노이즈 제거 (Noise Reduction)**
  - [ ] 필터링 (Gaussian, Median) (Filtering)
  - [ ] 비등방성 확산 (Anisotropic Diffusion), Non-local Means
  - [ ] 모달리티별 노이즈 특성 (Modality-Specific Noise)

- [ ] **이미지 정규화 및 표준화 (Image Normalization & Standardization)**
  - [ ] HU 단위 (CT 등) (Hounsfield Units)
  - [ ] Z-score, Min-Max 스케일링 (Z-score, Min-Max)
  - [ ] Intensity Clipping / Windowing

- [ ] **데이터 증강 (Data Augmentation)**
  - [ ] 회전, 이동, 크기 조정, 반전 (Rotation, Translation, Resizing, Flipping)
  - [ ] 밝기, 대비, 노이즈 추가 (Brightness, Contrast, Noise)
  - [ ] 3D 볼륨 증강 고려 (3D Volumetric Augmentation)
  - [ ] **Artifact/Noise 시뮬레이션**  
    - [ ] 가우시안 노이즈, 움직임 블러, 금속 아티팩트 등 인위 추가
    - [ ] 모델의 견고성을 높이기 위한 노이즈 주입 전략
  - [ ] **생성 모델 기반 증강 (Generative Model-based Augmentation)**  
    - [ ] GAN/확산 모델(DDPM 등)로 합성 의료 영상 생성
    - [ ] 데이터 부족이나 희귀 질환 케이스 증강 활용

---

### 6.3 분할 및 정합 (Segmentation & Registration)
- [ ] **의료 이미지 분할 (Medical Image Segmentation)**
  - [ ] 고전적 알고리즘 (Thresholding, Region Growing, Watershed)
  - [ ] U-Net, FCN, SegNet 등 (Deep Learning Segmentation)
  - [ ] 3D 볼륨 분할 (3D U-Net 등)
  - [ ] 파인튜닝, 전이 학습 (Fine-Tuning, Transfer Learning)

- [ ] **이미지 정합 (Image Registration)**
  - [ ] 강체 (Rigid), 선형 (Affine) 변환
  - [ ] 비선형 (Deformable) 정합 (B-spline, Demons 등)
  - [ ] 다중 모달리티 정합 (CT-MR 등)
  - [ ] 정합 품질 평가 (TRE, Dice 등)

- [ ] **다중 모달리티 융합 (Multi-Modality Fusion)**
  - [ ] CT+PET, MRI+PET 등 (CT-PET, MRI-PET Fusion)
  - [ ] 해부학적+기능적 영상 결합 (Anatomical + Functional Fusion)
  - [ ] 시각화 기법 (Fusion Visualization)

---

### 6.4 진단 응용 및 임상 검증 (Diagnostic Applications & Clinical Validation)
- [ ] **병변 검출 (Lesion Detection)**
  - [ ] 종양, 결절, 병변 검출 (Tumor, Nodule Detection)
  - [ ] CAD (Computer-Aided Detection) 시스템
  - [ ] FROC, Sensitivity, Specificity 등 (FROC, Sensitivity, Specificity)

- [ ] **종양 분류 (Tumor Classification)**
  - [ ] 양성/악성, 병기 (Benign/Malignant, Staging)
  - [ ] 방사선학적 특징 (Radiomics) + AI (Radiomics + AI)
  - [ ] 임상 지표 연동 (Clinical Biomarkers)

- [ ] **뇌 영상 분석 (Brain Imaging Analysis)**
  - [ ] 뇌 영역 분할 (Gray/White Matter, Ventricles 등)
  - [ ] 알츠하이머, 파킨슨 등 질환 (Alzheimer’s, Parkinson’s)
  - [ ] fMRI, DTI 등 특수 영상 (fMRI, DTI)

- [ ] **임상 데이터 평가 및 규제 (Clinical Validation & Regulations)**
  - [ ] 임상 시험 단계 (Clinical Trial Phases)
  - [ ] FDA, CE 등 의료기기 규제 (Medical Device Regulations)
  - [ ] 개인정보 보호 (HIPAA, GDPR 등)

---

### 6.5 데이터 주석 (Annotation)
- [ ] **의료 이미지 라벨링 도구 (Medical Image Labeling Tools)**
  - [ ] 3D Slicer, ITK-Snap, MITK, MONAI Label 등
  - [ ] 세그멘테이션 마스크, ROI 주석 (Masks, ROI Annotations)
  - [ ] 다기관 협업 워크플로우 (Multi-site Collaboration)

- [ ] **전문가 협업 및 검증 (Expert Collaboration & Validation)**
  - [ ] 의사/방사선사 협업 (Clinician Collaboration)
  - [ ] 주석 프로토콜 수립 (Annotation Protocol Establishment)
  - [ ] 라벨 품질 관리 (Dice, Inter-rater Reliability)

- [ ] **데이터 관리 및 보안 (Data Management & Security)**
  - [ ] PACS, RIS 시스템 연동 (PACS, RIS Integration)
  - [ ] DICOM 비식별화 (De-identification)
  - [ ] 버전 관리, 백업 전략 (Versioning & Backup)

---

### 6.6 확장 주제 (Extended Topics in Medical Imaging)
- [ ] **연합 학습 (Federated Learning)**
  - [ ] 병원 간 데이터 공유 없이 모델 공동 학습 (Secure Multiparty Computation 등)
  - [ ] Differential Privacy 활용 (Privacy-Preserving Techniques)

- [ ] **프라이버시·보안 기법 (Privacy-Preserving Techniques)**
  - [ ] Homomorphic Encryption, Differential Privacy
  - [ ] 의료 데이터 특성 고려 보안 및 윤리 이슈

- [ ] **설명 가능 AI (XAI) & 불확실성 추정 (Uncertainty Estimation)**
  - [ ] Grad-CAM, Integrated Gradients, SHAP 등
  - [ ] Bayesian Deep Learning, Monte-Carlo Dropout

- [ ] **도메인 일반화 & 도메인 적응 (Domain Generalization & Domain Adaptation)**
  - [ ] 다양한 병원/장비 간 도메인 시프트 극복
  - [ ] CT ↔ MRI, 3D ↔ 2D 변환 적용

- [ ] **약·반 지도학습 (Weakly/Semi-Supervised) 및 자가 지도학습 (Self-Supervised)**
  - [ ] 제한된 라벨 상황에서의 활용 (Consistency Regularization 등)
  - [ ] 라벨링 비용 절감을 위한 방법론

- [ ] **고급 영상 재구성 (Advanced Image Reconstruction)**
  - [ ] 저선량 CT 복원, 압축 센싱 (Compressed Sensing)
  - [ ] 모션 아티팩트 제거, 물리 모델+딥러닝 결합

- [ ] **Radiomics & Radiogenomics**
  - [ ] 방사선학적 특징 (형태, 텍스처 등)과 유전체 정보 결합
  - [ ] 다중 모달 해석 (영상 + 임상 데이터 + 유전체)

- [ ] **임상 워크플로우 & 규제 연동 (Clinical Workflow & Regulatory Integration)**
  - [ ] PACS/RIS 연동, FDA/CE 인증 프로세스
  - [ ] 임상 시험 단계 및 병행 연구 (Phase I~IV)

- [ ] **Multi-Task Learning & Active Learning**
  - [ ] 분할, 검출, 분류 동시 학습
  - [ ] 전문가 주석 필요 데이터 선별 (Active Learning)

- [ ] **Synthetic Data Generation**
  - [ ] GAN/Diffusion 기반 합성 의료 데이터 생성
  - [ ] 데이터 부족 및 프라이버시 문제 해소

---

## 7. Image-to-Image Translation

### 7.1 기본 개념 (Fundamental Concepts)
- [ ] **이미지 변환 정의 (Definition of Image-to-Image Translation)**
  - [ ] 픽셀 단위 변환 vs 의미론적 변환 (Pixel-level vs Semantic-level)
  - [ ] 도메인 변경, 스타일 전이 등 다양한 목적 (Purpose)

### 7.2 연구 현황 (Research Trends)
- [ ] **대표 기법 (Representative Approaches)**
  - [ ] 초기 AE/VAE 기반 방법 (Early Autoencoder/VAE Methods)
  - [ ] GAN 기반 (CycleGAN, Pix2Pix 등)
  - [ ] 최근 트랜스포머, 디퓨전 모델 (Transformers, Diffusion Models)

- [ ] **응용 분야 (Applications)**
  - [ ] 도메인 간 변환 (Domain Adaptation)
  - [ ] 스타일 전이 (Style Transfer)
  - [ ] 의료 영상 모달리티 변환 (예: CT ↔ MRI)

### 7.3 이미지 변환 모델의 종류 (Types of Image Translation Models)
- [ ] **주요 모델 분류 (Key Model Categories)**
  - [ ] Pix2Pix 계열, CycleGAN 계열, U-GAT-IT
  - [ ] VAE-GAN, 순수 GAN 기반
  - [ ] 디퓨전, 트랜스포머 기반

- [ ] **모델 선택 기준 (Model Selection Criteria)**
  - [ ] 성능 (화질, 변환 품질 등)
  - [ ] 학습 자원 요구사항
  - [ ] 데이터셋 크기 및 도메인 특성

- [ ] **평가 지표 (Evaluation Metrics)**
  - [ ] SSIM, PSNR, LPIPS, FID
  - [ ] 주관적 평가 vs 수치적 지표

### 7.4 확장 주제 (Extended Topics in Image-to-Image Translation)
- [ ] **Paired vs. Unpaired Translation**
  - [ ] Pix2Pix (쌍 데이터) vs CycleGAN (비쌍 데이터)
  - [ ] 의료영상의 정합 어려움 고려

- [ ] **Cycle Consistency & Attention/Transformer 기반 I2I**
  - [ ] CycleGAN의 Cycle Consistency 활용
  - [ ] Attention, Diffusion, Score-based 접근

- [ ] **Super-Resolution / Enhancement**
  - [ ] 저해상도 의료 영상의 고해상도화
  - [ ] Noise Reduction, Artifact Removal

- [ ] **등록(Registration) + 변환(Translation) 융합**
  - [ ] 3D 볼륨 기하학적 정합 후 도메인 변환 적용
  - [ ] 예: MR ↔ PET 정합 후 I2I

- [ ] **평가 지표 확장 (Medical-Specific Metrics)**
  - [ ] Dice, Hausdorff Distance, MAE (HU)
  - [ ] 임상적 유의성 반영

- [ ] **Real-Time Image Translation**
  - [ ] 실시간 변환을 위한 경량 모델 및 하드웨어 가속 (GPU/TPU)

- [ ] **Few-shot/Zero-shot Translation**
  - [ ] 극소량 또는 무라벨 데이터에서의 변환
  - [ ] 메타러닝, 도메인 적응 기법 결합

---

### 7.5 의료 영상 변환을 위한 단계별 구현 로드맵 (Step-by-Step Roadmap for Medical Image Translation)

1. **연구 및 문헌 조사 (Research & Literature Review)**  
   - [ ] 최신 의료 영상 변환 기법 탐색 (GAN, Diffusion, Transformer 등)  
   - [ ] Artifact/Noise 제거 및 증강 관련 연구 동향 조사  
   - [ ] 평가 지표(SSIM, PSNR, FID, Dice 등) 및 공개 데이터셋 파악

2. **문제 정의 및 데이터셋 준비 (Problem Definition & Dataset Preparation)**  
   - [ ] 저품질 → 고품질, CT ↔ MRI 등 변환 목적 구체화  
   - [ ] Artifact 포함 데이터와 깨끗한 참조(레이블) 확보 가능 여부 확인  
   - [ ] DICOM/NIfTI 전처리(정렬, 정규화 등) → 일관된 입력 구조 구성

3. **데이터 증강 전략 (Data Augmentation Strategy)**  
   - [ ] 전통적 증강(회전, 반전, 밝기, 대비 조정)  
   - [ ] Noise/Artifact 시뮬레이션(가우시안 노이즈, 금속 아티팩트 등)  
   - [ ] GAN/확산 모델 활용 합성 데이터 생성(희귀 케이스 보강)

4. **기본 모델 선택 및 구축 (Baseline Model Selection & Construction)**  
   - [ ] 짝지어진 데이터(Paired) → Pix2Pix / 비짝지어진 데이터(Unpaired) → CycleGAN 등  
   - [ ] PyTorch 기반 Baseline 구현(U-Net+GAN 등)  
   - [ ] 초기 실험을 통해 SSIM, PSNR 등 성능 확인

5. **Artifact 및 Noise 제거 기법 통합 (Integrating Artifact & Noise Removal)**  
   - [ ] 노이즈 제거 모듈/필터(FFT/Wavelet) 또는 보조 출력(Artifact Mask) 추가  
   - [ ] SSIM, 주파수 도메인 기반 추가 Loss 설계로 아티팩트 억제  
   - [ ] 멀티태스킹 또는 보조 목표(노이즈/Artifact 제거) 학습

6. **고급 모델 탐색 및 적용 (Advanced Model Exploration)**  
   - [ ] 다중 판별자, 멀티스케일 등 GAN 구조 개선  
   - [ ] 확산 모델(DDPM, Score-based)로 보다 안정적이고 높은 화질 시도  
   - [ ] Transformer 기반(Swin Transformer 등)으로 전역 컨텍스트 반영

7. **PyTorch 훈련 최적화 (Model Training & Optimization)**  
   - [ ] 하이퍼파라미터 튜닝(학습률, 배치 크기 등)  
   - [ ] AMP(Mixed Precision), 멀티-GPU, DDP 등으로 학습 가속  
   - [ ] 실시간 모니터링(TensorBoard), 체크포인트 관리

8. **평가 지표 설정 및 검증 (Evaluation & Validation)**  
   - [ ] 정량적 지표(SSIM, PSNR, FID, Dice 등) + 전문가 시각 평가 병행  
   - [ ] 공개 데이터셋에 대한 기존 연구와 동일 지표 비교  
   - [ ] 모델의 안정성(과적합, 재현성) 검증

9. **성능 비교 및 결과 분석 (Performance Comparison & Analysis)**  
   - [ ] Baseline 대비 성능 향상도 정리(표, 그래프)  
   - [ ] 최신 기법(SOTA)과의 정량/정성 비교  
   - [ ] 오블레이션 스터디로 각 모듈 기여도 평가

10. **모델 경량화 및 실시간 적용 (Model Compression & Real-Time Inference)**  
    - [ ] 가지치기, 양자화, 지식 증류로 모델 사이즈 축소  
    - [ ] PyTorch JIT/TorchScript, ONNX 등 사용하여 추론 최적화  
    - [ ] 실시간 변환 프로토타입(스트리밍 영상 → 변환 영상) 구현

11. **결론 및 향후 과제 (Conclusion & Future Work)**  
    - [ ] 연구 결과 종합 및 문서화(논문/보고서)  
    - [ ] 임상 적용 가능성 및 규제 이슈 검토  
    - [ ] 연합학습, 도메인 일반화, 추가 모달리티 확장 등 후속 연구 계획

---

## 8. 생성적 적대 신경망 (GAN)

### 8.1 기본 개념 (Basic Concepts)
- [ ] **GAN 구조 (GAN Architecture)**
  - [ ] 생성자 (Generator) vs 판별자 (Discriminator)
  - [ ] 미니맥스 게임 (Minimax Game)

- [ ] **학습 과정 (Training Process)**
  - [ ] 역전파 및 최적화 (Backprop & Optimization)
  - [ ] 모멘텀, Adam 등 (Momentum, Adam)

- [ ] **안정성 이슈 (Stability Issues)**
  - [ ] 모드 붕괴 (Mode Collapse)
  - [ ] 학습 불안정 (Gradient Exploding/Vanishing)

---

### 8.2 연구 현황 (Research Trends)
- [ ] **대표 변형/확장 모델 (Key Variants)**
  - [ ] cGAN, DCGAN (Conditional GAN, DCGAN)
  - [ ] WGAN, LSGAN, RaGAN (Loss Function Variants)
  - [ ] StyleGAN, BigGAN (High-Resolution Generation)

- [ ] **이미지 변환 적용 (Image Translation with GANs)**
  - [ ] Pix2Pix, CycleGAN, SPADE 등
  - [ ] 백본 아키텍처 (예: ResNet, U-Net)

---

### 8.3 GAN의 안정화 기법 (GAN Stabilization Techniques)
- [ ] **Loss Function Variants**
  - [ ] WGAN-GP (Gradient Penalty), LSGAN, Hinge Loss
  - [ ] Spectral Normalization

- [ ] **Mode Collapse 완화**
  - [ ] PacGAN, Minibatch Discrimination, Unrolled GAN
  - [ ] Label Smoothing, Two-Time Scale Update Rule (TTUR)

---

### 8.4 GAN의 의료 영상 적용 (Medical Imaging Applications of GANs)
- [ ] **데이터 증강 (Data Augmentation)**
  - [ ] 병변 합성 (Synthetic Lesion), 희귀 질환 데이터 증강
  - [ ] 조건부 합성을 통한 데이터 다양화 (Using cGAN)

- [ ] **이미지 복원 (Image Restoration)**
  - [ ] 노이즈 제거, 저선량 CT 복원
  - [ ] 아티팩트 제거 (금속, 모션 등)

- [ ] **병변 생성 (Lesion Synthesis)**
  - [ ] 모델 학습을 위한 라벨 확장
  - [ ] 임상 시뮬레이션 (가상 병변 생성)

- [ ] **모달리티 변환 (Modality Translation)**
  - [ ] MRI ↔ CT, PET ↔ CT 변환
  - [ ] Unpaired Translation 상황에서 CycleGAN 활용

- [ ] **Self-Supervised & Low-Label Scenarios**
  - [ ] 소량/무라벨 데이터 활용 (Feature Matching, Adversarial Loss)
  - [ ] 자기지도 학습 적용

---

## 9. Diffusion Models

### 9.1 기본 개념 (Basic Concepts)
- [ ] **디퓨전 모델 구조 (Diffusion Model Architecture)**
  - [ ] 순방향 과정 (노이즈 주입; Forward Process)
  - [ ] 역방향 과정 (이미지 복원; Reverse Process)
  - [ ] 마르코프 체인 (Markov Chain)

- [ ] **학습 방법 (Training Method)**
  - [ ] 변분 하한 (VLB) 최소화 (Minimizing Variational Lower Bound)
  - [ ] 노이즈 예측 네트워크 (Noise Prediction Network, 예: UNet 기반)

---

### 9.2 연구 현황 (Research Trends)
- [ ] **대표 기법 (Representative Methods)**
  - [ ] DDPM (Denoising Diffusion Probabilistic Models)
  - [ ] Score-based Generative Models
  - [ ] Latent Diffusion (예: Stable Diffusion)

- [ ] **응용 분야 (Applications)**
  - [ ] 고해상도 이미지 생성 (High-Resolution Image Generation)
  - [ ] Inpainting, Super-Resolution
  - [ ] 의료 영상 복원, 잡음 제거 (Medical Image Denoising/Restoration)
  - [ ] **Hybrid Approaches:** GAN과 디퓨전 모델 결합 가능성

---

## 10. Transformers

### 10.1 기본 개념 (Basic Concepts)
- [ ] **트랜스포머 구조 (Transformer Architecture)**
  - [ ] 셀프 어텐션 (Self-Attention)
  - [ ] 멀티 헤드 어텐션 (Multi-Head Attention)
  - [ ] 포지셔널 인코딩 (Positional Encoding)
  - [ ] 인코더-디코더 구조 (Encoder-Decoder)

- [ ] **학습 기법 (Training Techniques)**
  - [ ] 교차 엔트로피 손실 (Cross-Entropy Loss)
  - [ ] 마스킹 및 어텐션 매트릭스 (Masking & Attention Matrix)

---

### 10.2 연구 현황 (Research Trends)
- [ ] **자연어 처리 (NLP)**
  - [ ] BERT, GPT 시리즈 (BERT, GPT)
  - [ ] 대규모 사전학습 모델 (LLM)

- [ ] **컴퓨터 비전 (CV)**
  - [ ] 비전 트랜스포머 (ViT), Swin Transformer (ViT, Swin)
  - [ ] 이미지 분류, 객체 검출 등 (Image Classification, Detection)

- [ ] **멀티모달 (Multi-Modal)**
  - [ ] CLIP, Flamingo 등 (CLIP, Flamingo)
  - [ ] 이미지+텍스트 결합 (Image+Text Integration)
  - [ ] 기타 멀티모달 응용 (예: 음악, 영상 등)


---
published: true  
title: "[CNN] (2) CNN의 역사와 발전 과정, 주요 모델들"  
description: "CNN의 역사와 발전 과정, 주요 모델들을 살펴봅니다."  
header:  
  teaser: /assets/images/CNN/04_LeNet_5.png  
  og_image: /assets/images/CNN/04_LeNet_5.png  
  image_description: "CNN"  
date: 2025-02-15  
last_modified_at: 2025-02-17 21:00:00
toc: true  
toc_sticky: true  
use_math: true  
categories:  
  - CNN  
tags:  
  - DeepLearning  
  - CNN  
  - History  
---
이번 포스팅에서는 CNN의 역사와 발전 과정, 주요 모델들을 살펴봅니다.

## CNN 시리즈
1. [Convolutional Neural Networks 개요]({{ site.url }}{{ site.baseurl }}/cnn/1_CNN-Basics/)
2. [CNN의 역사와 발전 과정, 주요 모델들]({{ site.url }}{{ site.baseurl }}/cnn/2_CNN-History/)
3. [컨볼루션 레이어(Convolutional Layer)]({{ site.url }}{{ site.baseurl }}/cnn/3_Convolution-Layer/)
4. [패딩(Padding)과 스트라이드(Stride)]({{ site.url }}{{ site.baseurl }}/cnn/4_Padding-and-Stride/)
5. [풀링 레이어(Pooling Layer)]({{ site.url }}{{ site.baseurl }}/cnn/5_Pooling-Layers/)
6. [정규화 레이어(Normalization Layer)]({{ site.url }}{{ site.baseurl }}/cnn/6_Normalization-Layer/)
<!-- 7. [컨볼루션 레이어 쌓기(Stacked Convolutional Layers)]({{ site.url }}{{ site.baseurl }}/cnn/7_Stacked-Convolution-Layers/) -->


## 1. 생물학적 영감 (1950~1960년대)
### 1.1. **Hubel & Wiesel의 연구:**  
  1959년, Hubel과 Wiesel은 고양이의 시각 피질을 연구하면서, 뉴런들이 국소적 영역(수용영역)을 통해 시각 정보를 처리하며 계층적 구조를 형성한다는 사실을 밝혀냈습니다[^2].  
  - 이 발견은 후에 CNN의 **국소 연결(Local Connectivity)**과 **계층적 특징 추출(Hierarchical Feature Extraction)** 개념에 중요한 영감을 제공했습니다.

![Hubel & Wiesel]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/01_hubel_wissel_cat.png)
*In the classic neuroscience experiment, Hubel and Wiesel discovered a cat's visual cortex neuron(right) that ﬁres strongly and selectively for a light bar (left) when it is in certain positions and orientations[^11]*

## 2. Neocognitron의 등장 (1979년)
### 2.1. **Neocognitron (Fukushima, 1980):**  
  후쿠시마 쿠니히코는 **Neocognitron**을 제시하며, 지도학습 없이도 위치 변화에 강인한(shift invariant) 패턴 인식 메커니즘을 선보였습니다[^3].  
  - 이 모델은 **S-셀(단순세포)**과 **C-셀(복합세포)**로 구성된 계층적 구조를 사용하여, 후에 LeNet 등 초기 CNN 모델의 기초를 마련하는 데 큰 영향을 주었습니다.

![Neocognitron]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/02_Neocognitron.png)
*Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position[^3]*

## 3. 초기 CNN 실험과 LeNet (1980년대~1990년대)
### 3.1. **역전파 도입**  
  1986년 Rumelhart 등으로부터 **역전파(Backpropagation)** 알고리즘이 제안되면서, 신경망에 지도 학습을 적용할 수 있게 되었습니다[^4].  
  ![Neocognitron]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/03_Backpropagation.png)
  *Backpropagation: A method for finding the derivatives of the parameters of a neural network with respect to the loss function[^4]*
  
### 3.2. **LeNet-5 등장**  
  얀 르쿤과 동료들은 LeNet-5를 비롯한 초기 CNN 모델을 개발하여, 손글씨 숫자 인식(MNIST) 등 실용적 문제에서 탁월한 성능을 입증했습니다[^1].  

  - LeNet-5는 컨볼루션, 풀링, 완전연결층으로 구성되어 초기 응용 분야(예: 우편번호, 수표 인식)에서 널리 사용되었습니다.
  
  ![LeNet-5]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/04_LeNet_5.png)
  *LeNet-5: Architecture of LeNet-5, a Convolutional Neural Network for digits recognition. Each plane is a feature map, i.e., a set of units whose weights are constrained to be identical[^1]*


## 4. 딥러닝 혁명: AlexNet (2012년)
### 4.1. **AlexNet의 혁신:**  
  2012년, Krizhevsky, Sutskever, Hinton 등이 발표한 **AlexNet**은 5개의 컨볼루션층과 3개의 완전연결층으로 구성되어 GPU를 통한 대규모 학습을 가능하게 했습니다[^5].  
  - **주요 기여:**  
    - **GPU 가속:** 대규모 연산을 병렬 처리하여 깊은 네트워크의 학습을 실현.  
    - **ReLU 활성화 함수:** 기울기 소실 문제를 완화하여 빠른 학습을 지원.  
    - **드롭아웃:** 과적합을 줄이기 위한 정규화 기법 도입.  
  - 이 모델은 ImageNet 분류에서 기존 방법들을 압도하는 성능을 보여, 딥러닝 시대의 문을 열었습니다.

![AlexNet]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/05_AlexNet.png)
*AlexNet: An illustration of the architecture of our CNN, explicitly showing the delineation of responsibilities
between the two GPUs. One GPU runs the layer-parts at the top of the figure while the other runs the layer-parts
at the bottom. The GPUs communicate only at certain layers. The network’s input is 150,528-dimensional, and
the number of neurons in the network’s remaining layers is given by 253,440–186,624–64,896–64,896–43,264–
4096–4096–1000.[^5]*


## 5. 더 깊고 효율적인 모델들: VGG, GoogLeNet, ResNet (2014~2015년)
### 5.1. **VGGNet (2014):**  
  Simonyan과 Zisserman은 VGGNet에서 3×3 필터를 반복적으로 쌓아 깊은 네트워크를 구성하며, 네트워크 깊이가 성능에 미치는 영향을 명확하게 입증했습니다[^6].  
  - 단점으로는 파라미터 수가 많아 계산 비용과 메모리 요구량이 증가한다는 점이 있습니다.

![VGGNet]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/06_VGGNet.png)
*VGGNet: Variant of VGG Network[^6]*

### 5.2. **GoogLeNet (Inception v1, 2014):**  
  Szegedy 등은 **인셉션(Inception) 모듈**을 도입하여 1×1, 3×3, 5×5 필터를 병렬로 사용하면서 전체 파라미터 수를 대폭 줄이는 효율적인 구조를 제안했습니다[^7].  
  - 22계층의 깊이를 유지하며 글로벌 평균 풀링을 통해 최종 분류 성능을 향상시켰습니다.

![GoogLeNet]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/07_GoogLeNet.png)

*GoogLeNet: : GoogLeNet network with all the bells and whistles.[^7]*

### 5.3. **ResNet (2015):**  
  He 등은 **잔차 학습(Residual Learning)**과 **스킵 연결(Skip Connection)**을 도입하여, 152층 이상의 초심층 네트워크도 안정적으로 학습할 수 있도록 했습니다[^8].  
  - 이로 인해 매우 깊은 네트워크에서도 기울기 소실 문제를 극복하고 뛰어난 성능을 달성할 수 있었습니다.

![ResNet]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/08_ResNet.png)
*ResNet:  Example network architectures for ImageNet. 
Left: the VGG-19 model (19.6 billion FLOPs) as a reference. 
Middle: a plain network with 34 parameter layers (3.6 billion FLOPs).
Right: a residual network with 34 parameter layers (3.6 billion FLOPs). 
The dotted shortcuts increase dimensions.[^8]*

## 6. 최신 동향: EfficientNet, Vision Transformer 및 그 외 (2018년 이후)
### 6.1. **EfficientNet (2019):**  
  Tan과 Le는 NAS(Neural Architecture Search)와 **컴파운드 스케일링(Compound Scaling)** 기법을 도입하여, 네트워크의 깊이, 너비, 해상도를 균형 있게 확장하는 EfficientNet 모델을 발표했습니다[^9].  
  - EfficientNet-B7은 적은 파라미터(66M)로도 최첨단 ImageNet 성능(Top-5 약 97.1%)을 달성하여 효율성과 정확도를 모두 잡았습니다.

![EfficientNet]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/09_EfficientNet.png)
*EfficientNet: Model Scaling. (a) is a baseline network example; (b)-(d) are conventional scaling that only increases one dimension of network width, depth, or resolution. (e) is our proposed compound scaling method that uniformly scales all three dimensions with a ﬁxed ratio.[^9]*

### 6.2. **Vision Transformer (2021):**  
  Dosovitskiy 등은 Transformer를 이미지 분류에 직접 적용한 **Vision Transformer (ViT)**를 제안했습니다[^10].  
  - 이 모델은 이미지를 작은 패치로 분할한 후, 각 패치를 토큰처럼 다루어 전역적인 관계를 어텐션 메커니즘으로 학습합니다.

![ViT]({{ site.url }}{{ site.baseurl }}/assets/images/CNN/10_ViT.png)
*ViT: Model overview. We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. In order to perform classification, we use the standard approach of adding an extra learnable “classification token” to the sequence.[^10]*

## 7. 주요 CNN 아키텍처 비교

아래 표는 CNN 발전사의 주요 모델들을 요약하여, 각 모델의 구성, 파라미터 수, 성능 및 장단점을 비교한 내용입니다.

| **모델 (연도)**         | **주요 특징**                                                                                                                                      | **파라미터 수**         | **ImageNet Top-5 정확도**             | **장단점**                                                                                          |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|-------------------------------------|---------------------------------------------------------------------------------------------------|
| **LeNet-5 (1998)**  | 2개의 컨볼루션+풀링 층, 2개의 완전연결 층 등 총 7계층; 주로 손글씨 숫자 인식에 사용됨                                                                       | 약 60,000개         | MNIST에서 약 99% (ImageNet 적용X)      | **장점:** 초기 실용적 CNN, 간단한 구조<br>**단점:** 복잡한 이미지 인식에는 한계                     |
| **AlexNet (2012)**  | 5개의 컨볼루션 층 + 3개의 완전연결 층; ReLU, 드롭아웃, 데이터 증강 기법 도입; GPU를 통한 대규모 학습                                                                   | 약 60백만 개         | 약 84.7% (Top-5, 15.3% 오류율)          | **장점:** ImageNet 분류에서 획기적 성능 향상<br>**단점:** 모델 크기와 과적합 위험                     |
| **VGGNet (2015)**   | 16~19개의 컨볼루션 층과 3개의 완전연결 층; 3×3 필터를 반복적으로 사용하여 깊은 네트워크 구성 – 단, 층이 깊어 파라미터 수와 연산량이 크게 증가                                          | 약 138백만 개        | 약 92.7% (Top-5, 7.3% 오류율)           | **장점:** 단순한 구조로 깊이의 효과 입증<br>**단점:** 계산 비용과 메모리 요구량이 높음                   |
| **GoogLeNet (2014)**| Inception 모듈 도입 – 1×1, 3×3, 5×5 필터를 병렬로 적용; 22계층; 글로벌 평균 풀링 사용                                                                             | 약 6.8백만 개         | 약 93.3% (Top-5, 6.7% 오류율)           | **장점:** 효율적인 파라미터 사용과 높은 성능<br>**단점:** 모듈 구조의 복잡성                           |
| **ResNet (2015)**   | 잔차 학습(Residual Learning)을 통한 스킵 연결 도입; 152층 이상의 초심층 네트워크 구현; 주로 3×3 컨볼루션 사용                                                                  | 약 60백만 개         | 약 95.5% (Top-5)                      | **장점:** 극심한 깊이에서도 안정적 학습, 우수한 성능<br>**단점:** 네트워크 깊이에 따른 계산량 증가         |
| **EfficientNet (2019)**| NAS와 복합 스케일링(Compound Scaling) 기법을 통해 깊이, 너비, 해상도를 균형 있게 확장; EfficientNet-B7은 적은 자원으로 최첨단 성능 달성                                        | 약 66백만 개         | 약 97.1% (Top-5)                      | **장점:** 적은 자원으로 최첨단 성능 달성<br>**단점:** 설계 및 튜닝 복잡성                              |

*표: CNN의 주요 모델들을 요약한 비교표 (ImageNet 기준)*


## 8. CNN의 응용 분야

- **이미지 분류:** 수천 개의 카테고리를 학습하여 사진 속 객체를 자동으로 분류합니다. (예: 소셜 미디어 태깅, 이미지 검색)
- **객체 검출 및 위치 인식:** 이미지 내 객체의 위치를 정확히 파악하여 경계 상자를 생성합니다. (예: 자율주행, 감시 시스템)
- **의료 영상 분석:** CT, MRI, X-ray 등 의료 영상에서 종양이나 병변을 탐지하여 진단 보조에 활용됩니다.
- **자율주행 및 로봇 비전:** 차량 및 로봇에 장착된 카메라 영상을 실시간으로 분석하여 도로, 차선, 보행자 등을 인식합니다.
- **얼굴 인식 및 생체 인증:** 얼굴 특징을 학습해 보안 시스템 및 스마트폰 얼굴 인식에 사용됩니다.
- **예술 및 스타일 전이:** CNN 기반 **신경 스타일 전이(Neural Style Transfer)** 기법을 통해 한 이미지의 스타일을 다른 이미지에 적용하는 등 창의적 응용이 이루어집니다.

## 9. CNN의 한계

- **과적합과 데이터 요구량:** 파라미터 수가 많아 충분한 데이터가 없으면 과적합 위험이 있으며, 데이터 증강, 드롭아웃, 정규화 기법으로 이를 완화합니다.
- **계산 비용:** 깊은 네트워크는 학습 및 추론 시 많은 연산량과 메모리를 필요로 하며, GPU/TPU와 모델 압축 기법 등이 사용됩니다.
- **적대적 공격:** 미세한 노이즈에도 민감하여 악의적 조작 이미지에 오인식할 수 있는 취약점이 존재합니다.
- **해석 가능성:** 모델의 "블랙박스" 특성으로 인해 결정 과정 해석이 어려워, Grad-CAM 등 해석 기법이 연구되고 있습니다.

## 10. 연구 근황

- **CNN과 Transformer의 융합:** Vision Transformer(ViT)와 같은 Transformer 기반 모델 등장에 따라, CNN의 국소적 특징 추출과 Transformer의 전역적 관계 모델링을 결합한 하이브리드 모델들이 연구되고 있습니다.
- **자기지도 학습:**  라벨이 없는 대량의 데이터를 활용해 CNN을 사전 학습하는 기법(예: SimCLR, MoCo)이 각광받으며, 적은 라벨로도 높은 성능을 발휘할 모델들이 개발되고 있습니다.
- **효율성 극대화:** 경량 모델(MobileNet, EfficientNet) 및 NAS, 모델 압축 기법 등을 통해 동일한 성능을 내면서도 자원 소모를 최소화하는 연구가 진행 중입니다.
- **다양한 도메인 확장:** CNN의 개념은 이미지뿐 아니라 영상, 음성, 자연어, 그래프 등 다양한 데이터 도메인으로 확장되어 범용적인 특성 추출 도구로 자리 잡을 것입니다.

---

## 참고문헌

[^1]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.  
[^2]: Hubel, D. H., & Wiesel, T. N. (1959). Receptive fields of single neurones in the cat's striate cortex. *The Journal of Physiology*, 148(3), 574-591.  
[^3]: Fukushima, K. (1980). Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. *Biological Cybernetics*, 36, 193-202.  
[^4]: Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533-536.  
[^5]: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. In *Advances in Neural Information Processing Systems* (pp. 1097-1105).  
[^6]: Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. In *International Conference on Learning Representations*.  
[^7]: Szegedy, C., Liu, W., Jia, Y., et al. (2015). Going Deeper with Convolutions. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 1-9).  
[^8]: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 770-778).  
[^9]: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 6105-6114).  
[^10]: Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In *International Conference on Learning Representations*.
[^11]: NGUYEN, Anh; YOSINSKI, Jason; CLUNE, Jeff. Understanding neural networks via feature visualization: A survey. Explainable AI: interpreting, explaining and visualizing deep learning, 2019, 55-76.

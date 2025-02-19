---
published: false
title: "Noise Reduction Summary"
description: "Noise Reduction Summary"
header:
  teaser: /assets/images/noise_reduction_summary.png
  og_image: /assets/images/noise_reduction_summary.png
  image_description: "Noise Reduction Summary"
date: 2025-02-18
last_modified_at: 2025-02-18
toc: true
toc_sticky: true
use_math: true
categories:
  - NoiseReduction
tags:
  - ComputerVision
  - NoiseReduction
  - ImageProcessing
  - ImageDenoising
  - ImageFiltering
  - ImageRestoration
  - ImageEnhancement
  - ImageSuperResolution
---

<!-- [//]: # (작성 양식) -->
<!-- [//]: # (# 마크다운: kramdown) -->
<!-- [//]: # (# 수식 예시 => 수식이 필요할 경우에만 작성:  $$y = f&#40;z&#41; = f\left&#40;\sum_{i=1}^n w_i x_i + b\right&#41;$$) -->
<!-- [//]: # (# 언어: 한글, 주요 단어는 영문과 같이 표시) -->
<!-- [//]: # (# 참고문헌: 실제 존재하는 논문들을 형식에 맞게 기재) -->

# 컴퓨터 비전 분야 영상 노이즈 제거 연구 동향

컴퓨터 비전에서 **영상 노이즈 제거 (Image Denoising)** 연구는 초기의 필터 기반 기법부터 최첨단 딥러닝 모델까지 지속적으로 발전해왔습니다. 아래에서는 주요 학회(CVPR, ICCV, NeurIPS 등) 및 저널에 발표된 권위 있는 논문들을 중심으로, 전통적 필터링 기법에서 현대의 CNN, GAN, Transformer 기반 방법론까지의 발전 흐름을 정리합니다. 각 접근법의 대표 연구와 기여도를 관련 논문의 내용과 함께 요약하고, 방법론별 장점과 한계를 비교합니다.

## 1. 전통적 필터 기반 노이즈 제거 기법 (Traditional Filtering)

초기의 노이즈 제거 기법들은 **이미지 필터링**을 통해 잡음을 억제하고 중요한 구조를 보존하고자 하였습니다. 이러한 고전적 방법들은 주로 수학적 모델이나 통계적 특성을 이용하여 설계되었으며, 비교적 이해하기 쉽고 해석이 명확하다는 장점이 있습니다. 대표적인 전통적 노이즈 제거 기법들은 다음과 같습니다:

- **Anisotropic Diffusion (비등방 확산, Perona & Malik 1990)**: 이미지를 연속적인 확산 과정으로 해석하여, 에지(경계) 부근에서는 확산을 억제하고 평탄한 영역에서는 확산을 진행시키는 PDE 기반 방법입니다 ([Anisotropic diffusion - Wikipedia](https://en.wikipedia.org/wiki/Anisotropic_diffusion#:~:text=In%20image%20processing%20%20and,Gaussian%20filter%2C%20where%20the%20width)). 이를 통해 중요한 에지 등 **영상의 내용물을 보존하면서** 노이즈를 제거할 수 있음을 보였습니다. Perona와 Malik의 고전적 연구는 선형 확산이 에지를 희석시키는 한계를 지적하고, 국소 이미지 기울기에 따라 확산 정도를 조절하는 **비등방성 확산**을 제안하였습니다 ([Anisotropic diffusion - Wikipedia](https://en.wikipedia.org/wiki/Anisotropic_diffusion#:~:text=In%20image%20processing%20%20and,Gaussian%20filter%2C%20where%20the%20width)).

- **Bilateral Filtering (양방향 필터, Tomasi & Manduchi 1998)**: **비선형 에지-보존 평활화 필터**로, 공간적으로 가까운 픽셀일수록 그리고 밝기 값이 비슷할수록 큰 가중치를 부여하여 평활화를 수행합니다 ([ieee-paper.dvi](https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf#:~:text=Bilateral%20filtering%20smooths%20images%20while,Lab%20color)). 즉, **공간적 근접도(geometry)와 픽셀 값 유사도(photometric similarity)**를 동시에 고려하여 주변 픽셀들을 가중 평균함으로써, 에지 영역에서는 필터링 영향이 줄어들어 경계선을 유지하면서 잡음을 제거할 수 있습니다 ([ieee-paper.dvi](https://users.soe.ucsc.edu/~manduchi/Papers/ICCV98.pdf#:~:text=Bilateral%20filtering%20smooths%20images%20while,Lab%20color)). Bilateral 필터는 반복적 처리 없이 한 번의 필터링으로 수행되는 장점이 있으며, 컬러 영상에도 적용되어 **에지 부근의 색 번짐 현상(phantom color)**을 억제할 수 있음을 보였습니다.

- **Wavelet Thresholding (웨이블렛 임계치 기반 제거, Donoho 1995 등)**: **주파수 영역 또는 변환 영역**에서의 노이즈 제거 접근법입니다. 이미지를 웨이블렛 등 변환을 통해 주파수 성분으로 분해한 뒤, 노이즈에 해당하는 고주파 성분을 임계치(threshold)로 **조정(shrinkage)**하여 제거하는 방식입니다 ([](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf#:~:text=%28Yaroslavsky%20,Donoho%20%5B5%2C%204)). 이러한 방법은 수학적으로 잡음이 특정 계수에 미치는 영향을 분석하여, 작은 계수(대개 노이즈로 간주)를 제거하거나 감소시켜 노이즈를 제거하면서도 중요한 세부 신호는 유지합니다 ([](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf#:~:text=%28Yaroslavsky%20,Donoho%20%5B5%2C%204)). Donoho의 연구는 **VisuShrink** 등 간단한 임계치 기법으로도 효과적인 노이즈 억제가 가능함을 보여주어 이후 여러 변형된 웨이블렛 기반 기법의 토대를 마련했습니다.

- **Non-Local Means (비국소 평균, Buades et al. 2005)**: **이미지 자가-유사성(self-similarity)** 개념을 활용한 획기적인 기법으로, **전체 이미지에서 유사한 패치(조각)**들을 찾아 해당 픽셀 값을 평균내는 방식입니다 ([](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf#:~:text=the%20denoised%20value%20at%20x,2)). 국소 영역의 픽셀들만 사용하는 기존 필터와 달리, NL-Means는 이미지 전역에서 패치 단위로 유사성을 비교함으로써 보다 풍부한 정보를 활용합니다. Buades 등은 “한 픽셀의 값은 이미지 전체에서 이 픽셀 주변 패치와 모양이 비슷한 모든 픽셀들의 값으로부터 평균될 수 있다”는 원리를 제시했고 ([](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf#:~:text=the%20denoised%20value%20at%20x,2)), 이를 통해 **반복적이지 않으면서도(non-iterative)** 에지와 텍스처를 잘 보존하는 노이즈 제거가 가능함을 보였습니다. NL-Means는 이후 나오는 비국소적 방법들의 기반이 되었으며, 국소 필터 대비 **세부 정보 보존에 큰 강점**을 보였습니다.

- **BM3D (Block-Matching and 3D filtering, Dabov et al. 2007)**: 유사한 패치들을 모아 **3차원 스택을 구성**하고, 해당 3D 배열에 대한 **공동 필터링(collaborative filtering)**을 수행하는 기법입니다 ([An Analysis and Implementation of the BM3D Image Denoising Method](https://www.ipol.im/pub/art/2012/l-bm3d/article_lr.pdf#:~:text=BM3D%20is%20a%20recent%20denoising,We%20hope%20this%20new)). 구체적으로, 잡음이 낀 영상에서 서로 비슷한 2D 패치들을 블록 매칭으로 찾아 3D 블럭을 만든 후, 3D 변환(domain)에서 임계치 필터링(계수 shrinkage)을 적용하고 역변환하여 노이즈를 제거합니다 ([An Analysis and Implementation of the BM3D Image Denoising Method](https://www.ipol.im/pub/art/2012/l-bm3d/article_lr.pdf#:~:text=BM3D%20is%20a%20recent%20denoising,We%20hope%20this%20new)). 이렇게 필터링된 패치들을 원래 위치에 재배치하고 평균(Aggregation)하여 최종 이미지를 복원합니다. BM3D는 **비국소적 패치 유사성 + 변환 영역 희소성(sparsity)**을 결합하여 당대 최고 수준의 성능을 달성했으며, 특히 **가우시안 잡음 제거의 정석적인 최첨단(state-of-the-art)** 방법으로 오랫동안 평가되었습니다 ([An Analysis and Implementation of the BM3D Image Denoising Method](https://www.ipol.im/pub/art/2012/l-bm3d/article_lr.pdf#:~:text=BM3D%20is%20a%20recent%20denoising,We%20hope%20this%20new)). 실제로 “2007년에 제안된 BM3D는 이후 여러 해 동안 AWGN 제거의 최첨단 기법으로 자리매김했다”는 평가를 받고 있습니다 ([JeffOwOSun/gpu-bm3d: GPU-accelerated denoising with ... - GitHub](https://github.com/JeffOwOSun/gpu-bm3d#:~:text=GitHub%20github,three%20main%20stages%3A%20Block)). BM3D의 성공으로, 유사 패치 그룹화와 공동 필터링이라는 개념이 이미지 복원 분야 전반에 큰 영향을 주었습니다.

이외에도 K-SVD(Aharon et al., 2006)와 같은 **사전 학습(Dictionary Learning)** 기반 희소 표현 기법, Fields of Experts(Roth & Black, 2005)와 같은 **Markov Random Field** 학습 기법 등도 제안되었으나, 전통적으로는 위에서 언급한 **에지-보존 필터**와 **비국소적(self-similar) 필터**들이 가장 널리 활용되었습니다. 전통 기법들은 **이해하기 쉽고 계산이 비교적 빠르며** 별도의 훈련 데이터가 필요 없다는 장점이 있지만, 복잡한 패턴의 재현이나 고급 통계 구조 학습에는 한계가 있습니다. 예를 들어 텍스처까지 살리면서 노이즈를 제거하는 데에는 제약이 있고, **노이즈 유형에 최적화**하기 위해 수작업으로 알고리즘을 조율해야 했습니다.

## 2. CNN 기반 딥러닝 노이즈 제거 (CNN-Based Denoising)

딥러닝의 발전과 함께 2010년대 중반부터 **합성곱신경망(CNN)**이 이미지 노이즈 제거 분야에 본격 도입되어 획기적인 성능 향상을 이루었습니다. 초기에는 비교적 얕은 신경망이나 자동인코더를 이용한 시도가 있었고, 이후 매우 깊은 CNN 구조와Residual Learning 등이 도입되면서 전통 기법들의 성능을 크게 앞지르게 되었습니다. 주요 연구를 연대기와 방법 측면에서 살펴보면 다음과 같습니다:

- **초기 신경망 기반 시도**: 2012년 **Burger et al. (CVPR 2012)**의 연구는 “기존 BM3D 등의 알고리즘을 인간이 설계하는 대신, 다층 퍼셉트론(MLP)을 크게 구성하고 대량의 데이터로 학습시켜도 최고 성능을 낼 수 있다”는 것을 처음으로 입증했습니다 ([](https://people.tuebingen.mpg.de/burger/neural_denoising/files/neural_denoising.pdf#:~:text=and%20noise,a%20patch%20con%02tains%20enough)). 이 논문 *"Image Denoising: Can plain neural networks compete with BM3D?"* ([
		Image Denoising in Deep Learning: A Comprehensive Survey
							| Electrical Engineering Technical Journal
			](https://eetj.mtu.edu.iq/eetj/index.php/home/article/view/2#:~:text=Burger%2C%20H,6247952))에서, 노이즈 패치 입력에 대한 정답 패치를 출력으로 하는 단순한 MLP를 학습하여 BM3D와 견줄 만한 **Gaussian 노이즈 제거 성능**을 달성하였고, 이는 **딥러닝 기반 잡음 제거의 가능성**을 보여준 선구적인 결과입니다 ([](https://people.tuebingen.mpg.de/burger/neural_denoising/files/neural_denoising.pdf#:~:text=and%20noise,a%20patch%20con%02tains%20enough)). 같은 시기 **Xie et al. (NIPS 2012)**는 **Stacked Denoising Autoencoder**를 적용하여 이미지의 노이즈 제거와 빈 영역 채우기(inpainting)를 수행하였고, **멀티레이어 신경망이 데이터로부터 노이즈 패턴을 학습**할 수 있음을 시사하였습니다. 이러한 연구들은 비교적 얕은 네트워크였지만, **충분한 학습 데이터**와 **패치 기반 훈련**을 통해 전통 기법에 근접하거나 뛰어난 결과를 얻을 수 있음을 보였습니다.

- **Residual Learning 도입과 매우 딥한 CNN**: 딥러닝이 발전함에 따라 네트워크가 더욱 깊어졌고, 2015~2017년경 획기적인 CNN 기반 기법들이 등장했습니다. 특히 **Zhang et al.의 DnCNN (CVPR 2017/TIP 2017)** 모델은 **잔차 학습(Residual Learning)**과 **배치정규화(Batch Normalization)**를 통한 **20개 이상의 레이어**로 구성된 매우 깊은 CNN으로, **블라인드 가우시안 노이즈 제거**(노이즈 표준편차 미지의 상황)까지 하나의 모델로 처리할 수 있음을 보여주었습니다 ([[1608.03981] Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981#:~:text=attracting%20considerable%20attentions%20due%20to,is%20able%20to%20handle%20Gaussian)) ([[1608.03981] Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981#:~:text=method%20into%20image%20denoising,tasks%20such%20as%20Gaussian%20denoising)). DnCNN은 **노이즈를 직접 제거하는 대신 “노이즈 성분”을 예측하도록 학습(Residual)**함으로써 학습을 안정화하고 성능을 극대화하였습니다 ([[1608.03981] Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981#:~:text=attracting%20considerable%20attentions%20due%20to,is%20able%20to%20handle%20Gaussian)). 그 결과, 동일 모델 하나로 **다양한 잡음 수준의 AWGN 제거**뿐만 아니라 **이미지 초해상도**나 **JPEG 복원**까지 처리할 수 있음을 보여주어 모델 범용성이 향상되었고 ([[1608.03981] Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981#:~:text=method%20into%20image%20denoising,tasks%20such%20as%20Gaussian%20denoising)), 전통 기법인 BM3D 대비 크게 향상된 PSNR 성능 향상을 달성했습니다. DnCNN의 성공으로 **Residual Learning**은 이후 이미지 복원용 CNN의 사실상 표준 기법이 되었고, **딥러닝이 기존 최첨단 성능을 뛰어넘는 분기점**이 되었습니다.

- **다양한 CNN 아키텍처의 발전**: DnCNN 이후 많은 변형과 개선된 신경망 구조들이 제안되었습니다. **타이트하게 연결된 자동인코더 구조**인 RED-Net(Mao et al. 2016)이나 **대각선 연결(Dense Skip)**을 활용한 **Residual Encoder-Decoder 네트워크** 등이 이미지 복원에 활용되어, 보다 깊은 네트워크에서 **그레이디언트 소실 없이 특징을 전달**할 수 있게 했습니다. **MemNet(Tai et al. ICCV 2017)**은 **Persistent Memory Module**을 도입하여 장기적인 계층 간 정보를 보존함으로써 복원 성능을 높였고, **UNet** 구조를 차용한 대회성 네트워크들도 시도되었습니다. 한편, **Chen & Pock의 TNRD (Trainable Nonlinear Reaction Diffusion, 2015)**는 전통적인 PDE 기반 방법을 **네트워크 형태로 펼쳐(unroll)** 학습한 것으로, **소규모 파라미터로 BM3D에 필적하는 성능**을 보여주며 모델 기반과 학습 기반의 가교 역할을 하였습니다. **FFDNet(Zhang et al. TIP 2018)**은 **가변 잡음 수준 맵(noise level map)**을 입력으로 받아들임으로써 **하나의 모델로 다양한 잡음 세기에 대응**할 수 있고, 이미지를 다운샘플링하여 처리함으로써 **속도를 향상**시킨 실용적인 접근입니다 ([[1710.04026] FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising](https://arxiv.org/abs/1710.04026#:~:text=denoising%20convolutional%20neural%20network%2C%20namely,and%20real%20noisy%20images%20are)). 이처럼 CNN 기반 방법들은 레지듀얼 블록, 멀티-스케일 구조, dilated conv, 등 다양한 기법을 통해 성능과 효율을 개선해왔습니다. 특히 **Non-local Neural Networks** 개념도 도입되어 **자체Attention**으로 이미지 내 유사 패치를 탐색하는 **N3Net (2018)**이나, **Non-Local Recurrent Network (NLRN, 2018)** 등이 **비국소적 자기유사성**을 CNN에 통합하여 성능을 향상시켰습니다. 전반적으로 최근까지의 CNN 기반 기법들은 **풍부한 표준 데이터셋에 대한 지도 학습**을 통해, **전통 기법 대비 월등한 PSNR/SSIM 성능**과 깨끗한 복원 결과를 달성하고 있습니다.

- **실제 노이즈 및 블라인드 denoising**: 한편, 실제 사진에 존재하는 노이즈는 가우시안 분포가정과 달라서, **실세계 노이즈 처리**를 위한 연구도 활발합니다. **CBDNet (Guo et al. CVPR 2019)**은 **카메라 센서 노이즈 모델**을 학습하여 **블라인드 노이즈 제거**를 수행한 예이며, 이후 **SIDD**나 **DND**와 같은 실제 노이즈 데이터셋이 공개되면서 많은 네트워크들이 여기에 최적화되었습니다. 예를 들어, **Grouped Residual Dense Network (GRDN, 2019)**은 실영상 노이즈에 강인한 CNN 구조를 제안했고, **Kim et al. (CVPR 2020)**은 **Adaptive Instance Normalization**을 이용한 **도메인 변환 학습**으로 **합성-실제 노이즈 갭을 줄이기**도 했습니다 ([Heterogeneous window Transformer for image denoising](https://arxiv.org/html/2407.05709v2#:~:text=%2A%20%20%5B12%5D%20D.,Wang)). 이러한 기법들은 **카메라 이미지의 복잡한 잡음 특성(색변화, 감마왜곡 등)을 모델링**하거나, 합성 데이터로 학습한 모델을 실제 데이터에 미세조정(finetuning)하여 **일반화 성능**을 향상시킵니다. 다만 실제 노이즈의 ground truth를 얻기 어려워, 다음에 언급될 **비지도 학습 기법**이나 **GAN**을 활용하는 방향과도 접목되고 있습니다.

### ※ CNN 기반 접근의 요약 및 의의  
CNN 기반 방법들은 훈련 데이터만 충분하다면 **잡음 패턴의 복잡한 비선형 변환을 학습**하여 제거할 수 있으므로, **인간이 설계한 필터의 한계를 뛰어넘는 성능**을 달성했습니다. 특히 **딥한 네트워크**와 **Residual 학습** 기법은 노이즈 제거 성능을 큰 폭으로 끌어올려, 전통 기법 대비 1dB 이상의 PSNR 향상을 이루는 등 혁신적 변화를 가져왔습니다 ([](https://people.tuebingen.mpg.de/burger/neural_denoising/files/neural_denoising.pdf#:~:text=and%20noise,a%20patch%20con%02tains%20enough)) ([[1608.03981] Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981#:~:text=attracting%20considerable%20attentions%20due%20to,is%20able%20to%20handle%20Gaussian)). 다만 **지도 학습(supervised)**에 의존하므로 **깨끗한 정답 이미지가 필요한 제약**이 있고, 훈련된 모델은 **학습된 노이즈 분포에 특화**되는 경향이 있어 **다른 유형의 노이즈나 도메인으로 일반화**하는 데 한계가 있을 수 있습니다. 이러한 한계를 극복하기 위해, 최근에는 **비/약지도 학습**이나 **생성 모델**을 도입하는 방향으로 연구가 확장되고 있습니다.

## 3. GAN 및 자기지도 기법을 활용한 접근 (GAN and Self-Supervised Approaches)

CNN 기반 방법들은 뛰어난 성능을 보였지만, 대부분 **픽셀 단위의 손실함수(MSE 등)**를 사용하기 때문에 **과도한 평활화로 인한 세부 손실(oversmoothing)** 현상이 지적되었습니다. 또한 **정답 이미지가 없는 경우에는 학습이 어렵다는 한계**가 있었습니다. 이를 보완하고자, **적대적 생성망(GAN)**을 도입하여 보다 **사람이 보기 좋은(perceptual)** 결과를 얻거나, **자기 지도(self-supervised) 학습**을 통해 **노이즈 레이블 없이**도 모델을 학습시키는 방향의 연구가 진행되었습니다.

- **GAN을 통한 퍼셉추얼 품질 향상**: **Generative Adversarial Network (GAN)**은 한쪽(netG)이 이미지를 복원하고 다른 쪽(netD)이 복원된 이미지의 진위를 판별하는 **적대적 학습** 구조로, 초해상도 등에서 화질 개선에 사용된 것을 노이즈 제거에도 응용한 것입니다. 예를 들어, **Yang et al. (TMI 2018)**는 **Wasserstein GAN과 퍼셉튜얼 손실**을 활용하여 저선량 CT 이미지를 고화질로 복원하였고, 노이즈는 줄이면서도 **텍스처와 세부 구조를 더욱 현실감 있게 재현**함을 보였습니다 ([Heterogeneous window Transformer for image denoising](https://arxiv.org/html/2407.05709v2#:~:text=Recognition%20Workshops%2C%202019%2C%20pp,based%20noise%20model%20for)). 이처럼 GAN 기반 접근은 단순한 픽셀 정확도 대신 **영상의 시각적 품질을 향상**시키며, 결과 이미지가 **깨끗하면서도 날카로운 디테일**을 갖도록 유도합니다. 다만 GAN 특유의 학습 불안정성과 **가짜 세부묘사(hallucination)** 발생 가능성은 신중히 다뤄야 하는 부분입니다.

- **Unpaired Denoising (비짝지 데이터 학습)**: GAN을 이용하면 clean-noisy 이미지 쌍이 없더라도 학습이 가능하다는 점도 중요한 이점입니다. **Hong et al. (AAAI 2020)**의 연구는 **Conditional GAN**을 활용하여 **짝을 이루지 않은(unaligned) 잡음 영상들만으로** 깨끗한 영상을 생성하는 방법을 제안하였습니다 ([Heterogeneous window Transformer for image denoising](https://arxiv.org/html/2407.05709v2#:~:text=Vision%2C%202020.%20,1%E2%80%939)). 깨끗한 이미지와 노이즈 이미지를 별도로 입력 받아 adversarial하게 매칭시켜주는 기법으로, 일종의 **CycleGAN 원리**를 적용해 **노이즈 도메인 → 클린 도메인** 변환을 학습합니다. 이러한 **비지도 적대적 학습**을 통해, 현실에서는 동일 장면의 깨끗/노이즈 이미지 쌍을 얻기 어려운 경우에도 **모델 학습이 가능**해졌습니다. 유사하게, **GAN 기반 노이즈 모델링(Tran et al. ACCV 2020)** 연구에서는 깨끗한 이미지에 실제 노이즈 특성을 가진 GAN-생성 노이즈를 첨가하는 방식을 통해 데이터셋을 생성, **실제와 유사한 노이즈를 합성하여** 네트워크를 학습시키기도 했습니다 ([Heterogeneous window Transformer for image denoising](https://arxiv.org/html/2407.05709v2#:~:text=,Joint%20Conference%20on%20Neural%20Networks)). 이런 접근들은 **현실 세계 노이즈 분포 추정**이나 **비정합 데이터 학습** 문제를 GAN으로 해결한 예시들입니다.

- **자기지도(Self-Supervised) 노이즈 제거**: 2018년 **Noise2Noise (Lehtinen et al., ICML 2018)** 논문은 노이즈 제거 분야에 큰 반향을 일으켰는데, **깨끗한 이미지가 전혀 없이도 노이즈 이미지만으로 학습 가능**하다는 것을 보였습니다. 핵심 아이디어는 동일한 장면에 대한 **두 장의 서로 다른 노이즈 이미지 쌍**을 사용하면, 한 장을 입력으로 하고 다른 한 장을 타깃으로 삼아 학습하더라도 **최종 출력이 깨끗한 이미지에 수렴**한다는 것입니다 ([[1803.04189] Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189#:~:text=,based%20on%20noisy%20data%20only)). 저자들은 가우시안 노이즈뿐 아니라 몬테카를로 렌더링 노이즈, MRI undersampling 노이즈 등 다양한 잡음에 대해 이 기법을 적용했고, **깨끗한 레퍼런스 없이도 지도 학습한 경우에 맞먹는 성능을 달성**함을 보여주었습니다 ([[1803.04189] Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189#:~:text=,based%20on%20noisy%20data%20only)). Noise2Noise의 결과는 **“오직 잡음 투성이 데이터만으로도 깨끗한 영상을 복원할 수 있다”**는 개념적 단순하면서도 강력한 결론으로 요약됩니다 ([[1803.04189] Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189#:~:text=,based%20on%20noisy%20data%20only)). 이후 **Noise2Void (Krull et al., 2019)**와 **Noise2Self (Batson & Royer, 2019)** 등이 발전형으로 제안되어, **하나의 노이즈 이미지 내에서 자가완성(masking)을 통해 학습**하거나 **자체 통계로 자기지도**하는 방식으로 **단일 이미지로부터의 노이즈 제거**를 가능하게 했습니다. 이러한 자기지도 기법들은 **의료영상** 등 깨끗한 데이터 얻기 어려운 분야나, **실시간 어플리케이션** 등에서 유용하며, CNN의 활용 범위를 넓혔습니다.

### ※ GAN/자기지도 접근의 의의와 한계  
GAN 기반 방법들은 **인간 시각에 더욱 만족스러운 결과**를 생성할 수 있으며, **정합되지 않은 데이터셋으로도 학습**할 수 있다는 장점이 있습니다. 이를 통해 기존 지도학습 CNN의 제한을 일부 극복했지만, GAN 특유의 **훈련 난이도**와 **평가의 주관성(객관적인 지표 저하 가능)**은 고려해야 합니다. 자기지도 학습 기법은 **데이터 요구사항을 혁신적으로 낮추었다는 의의**가 있으며, **노이즈2노이즈** 패러다임 이후로 **Label 없이도 학습하는 Low-Level Vision** 연구가 활발해졌습니다. 다만 이러한 방법들은 **특정 가정(노이즈가 평균 0 등)**에 기반하므로 그 조건이 어긋나는 경우 성능이 저하될 수 있고, 완전히 깨끗한 결과를 얻기까지는 한계가 있을 수 있습니다. 따라서 최신 연구들은 자기지도 기법과 지도학습을 혼합하거나(GAN+Noise2Noise 등) Transformer와 같은 새로운 구조와 결합하여 한계를 극복하려 하고 있습니다.

## 4. Transformer 기반 최신 노이즈 제거 (Transformer-Based Approaches)

**Vision Transformer**의 등장은 이미지 노이즈 제거 분야에도 새로운 흐름을 가져왔습니다. Transformer는 원래 자연어 처리에서 도입된 **Self-Attention 메커니즘**을 기반으로 하는 모델로, **멀리 떨어진 위치 간의 상관관계**를 효과적으로 포착할 수 있다는 장점이 있습니다. 2020년대에 Vision Transformer들이 고해상도 이미지 처리에 응용되면서, 노이즈 제거에도 CNN을 넘어선 **글로벌한 문맥 정보 활용**이 시도되고 있습니다. 주요 연구를 살펴보면 다음과 같습니다:

- **IPT: 대규모 사전학습 Transformer (Chen et al., CVPR 2021)** – **Pre-Trained Image Processing Transformer (IPT)**는 **대용량의 이미지 복원 사전학습 모델**로서, 저자들은 ImageNet에서 수백만 개의 이미지를 이용해 노이즈가 추가된 입력-출력 쌍을 생성하고 이를 Transformer로 학습시켰습니다 ([[2012.00364] Pre-Trained Image Processing Transformer](https://arxiv.org/abs/2012.00364#:~:text=task%20%28e.g.%2C%20denoising%2C%20super,level%20benchmarks.%20Code%20is)). IPT 모델은 이렇게 **노이즈 제거, 초해상도, 비좁은(real) 비 오는 영상 제거 등의 여러 태스크를 멀티헤드로 동시에 학습**한 후 각 태스크별로 파인튜닝하는 방식으로 사용되며, 하나의 사전학습 모델로 당시 **각 태스크의 최첨단 성능을 경신**했음을 보고했습니다 ([[2012.00364] Pre-Trained Image Processing Transformer](https://arxiv.org/abs/2012.00364#:~:text=is%20trained%20on%20these%20images,URL%20and%20this%20https%20URL)). 특히 노이즈 제거의 경우, **대규모 데이터로 사전학습된 Transformer의 강력한 표현력**을 활용하여 기존의 특화된 CNN들을 능가하는 성능을 보였는데 ([[2012.00364] Pre-Trained Image Processing Transformer](https://arxiv.org/abs/2012.00364#:~:text=is%20trained%20on%20these%20images,URL%20and%20this%20https%20URL)), 이는 **모델 용량과 데이터 규모의 힘을 Low-Level Vision 문제에도 입증**한 결과로 평가됩니다. IPT는 Vision Transformer를 노이즈 제거에 본격 도입한 초기 연구로서, **Transformer가 CNN 대비 뛰어난 장거리 의존성 학습으로 미세한 구조 복원에 유리**함을 보여주었습니다.

- **효율적인 Transformer 구조들**: 순수 Transformer는 **자기어텐션 연산량이 이미지 해상도의 제곱에 비례**하여 고해상도 영상에는 비효율적일 수 있습니다. 이를 개선하기 위해 **국소 영역 윈도우 어텐션**이나 **채널 축 어텐션** 등을 도입한 변형들이 등장했습니다. **SwinIR (Liang et al., ICCV 2021)**은 Swin Transformer 구조를 이미지 복원에 적용한 것으로, 이미지를 작은 윈도우 단위로 self-attention을 수행하면서 윈도우를 이동(Shift)하여 전역 정보도 서서히 통합하는 방식입니다. SwinIR은 파라미터 수 대비 높은 성능을 보이며, 노이즈 제거 벤치마크에서 CNN 대비 competitive한 결과를 달성하였습니다 ([Heterogeneous window Transformer for image denoising](https://arxiv.org/html/2407.05709v2#:~:text=,1833%E2%80%931844)). **Restormer (Zamir et al., CVPR 2022)**는 **채널 기반 다중 Dconv 헤드 이송 어텐션(MDTA)**과 게이팅 기법을 통해 **자기어텐션의 계산 복잡도를 선형으로 줄이면서도** 전역 문맥을 포착하는 새로운 Transformer 블록을 제안했습니다 ([Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf#:~:text=image%20restoration%20that%20is%20capable,This%20is)) ([Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf#:~:text=work%2C%20we%20reformulate%20the%20first,forward%20and%20allows%20subsequent%20layers)). Restormer는 **인코더-디코더** 구조로 다중 해상도 특징을 처리하며, 고해상도 이미지의 노이즈 제거를 효율적으로 수행할 수 있음을 보였습니다 ([Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf#:~:text=divide%20the%20input%20image%20into,head)) ([Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf#:~:text=Apart%20from%20the%20above%20architectural,Restormer%20on%2016%20benchmark%20datasets)). 저자들은 Restormer가 **16개 이상의 복원 벤치마크(강우 제거, 모션 블러 제거, 포커스 블러 제거, 그리고 가우시안/실제 노이즈 제거 등)**에서 **최첨단 성능**을 달성했음을 보고하고 있는데 ([Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf#:~:text=comprehensive%20experiments%20and%20demonstrate%20state,architectural%20designs%20and%20experimental%20choices)), 이는 Transformer 기반 모델이 **다양한 저레벨 비전 문제에서 최고 성능을 낼 수 있음**을 입증합니다. 이외에도 **Uformer (Wang et al. 2021)**, **Attention-Augmented CNN** 등 CNN과 Transformer 아이디어를 절충한 하이브리드 모델들도 제안되어, **지역적 특징 추출은 CNN으로, 장거리 상호작용은 Transformer로 담당**시키는 방식으로 성능과 효율을 균형 있게 추구하고 있습니다 ([A Dynamic Network with Transformer for Image Denoising - MDPI](https://www.mdpi.com/2079-9292/13/9/1676#:~:text=This%20article%20aims%20to%20make,with%20Transformer%20for%20image)).

- **최신 경향**: 최신 Transformer 기반 방법들은 단순 성능 향상을 넘어 **경량화와 실시간 처리 가능성**에도 초점을 맞추고 있습니다. 예를 들어, **정밀도를 약간 희생하고도 경량화한 Transformer**나, **CNN기반의 MobileNet 스타일 경량 복원망에 어텐션 모듈을 삽입**하는 등 실용적 모델들도 연구되고 있습니다. 또한 **Diffusion Probabilistic Model**이나 **Score-based Model**과 같이 **확률적 생성 모델**을 노이즈 제거에 접목하려는 시도도 최근 5년 사이 등장하였습니다. 이러한 모델들은 **노이즈를 단계적으로 제거하며 이미지를 생성**하는 접근으로, 특히 **강한 노이즈나 복합적인 왜곡**이 있는 경우 혁신적인 성능을 보여줄 잠재력이 있다고 평가됩니다. 다만, Transformer 및 확률적 모델들은 여전히 **연산 비용과 메모리 사용량이 커서** 현실 애플리케이션에 적용하기 위해 최적화가 필요합니다. 

### ※ Transformer 접근의 장점과 한계  
Transformer 기반 모델들은 **Self-Attention을 통해 이미지 전역의 관계를 학습**하므로, **멀리 떨어진 반복 패턴이나 구조적 노이즈**를 제거하는 데 유리합니다. 또한 최근 모델들은 **CNN이 놓치기 쉬운 장거리 특징을 보완**하여, 텍스처나 구조 복원 면에서 뛰어난 성능을 보여주고 있습니다 ([Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf#:~:text=comprehensive%20experiments%20and%20demonstrate%20state,architectural%20designs%20and%20experimental%20choices)). 그러나 **막대한 데이터로의 사전학습**이 없으면 작은 데이터셋에서는 과적합 우려가 있고, 파라미터 수도 커지기 쉬워 **연산 자원**을 많이 요구합니다. 따라서 상대적으로 **데이터가 적거나 실시간성이 중요한 경우**, 여전히 CNN이나 경량화된 hybrid 모델이 선호될 수 있습니다. 요약하면, Transformer는 **성능 지향적인 최신 동향**으로 자리잡았지만 **실용적 제한사항**도 존재하며, 이를 해결하기 위한 연구가 병행되고 있습니다.

## 5. 접근 방법론별 비교 요약 (장점 및 한계)

마지막으로, 앞서 살펴본 다양한 노이즈 제거 방법들을 **방법론별로 주요 장점과 한계**를 비교하면 다음과 같습니다:

- **전통적 필터링 기법**:  
  - *장점:* 이론적 기반이 단순명료하여 **결과 해석이 용이**하고, 대체로 **연산이 빠르고 가벼움**니다. 사전 학습 없이도 적용 가능하며, 특정 애플리케이션에 **파라미터 튜닝으로 최적화**하기 쉬운 편입니다.  
  - *한계:* **데이터로부터 학습하는 능력이 없으므로 복잡한 패턴을 충분히 복원하지 못하고**, 잡음을 제거하려면 어느 정도 **디테일 손실(blur)**이 불가피합니다. 또한 각 기법이 가정하는 노이즈 모델에 따라 성능 편차가 크며, **여러 종류의 노이즈나 복합적인 왜곡에는 취약**합니다.

- **CNN 기반 방법**:  
  - *장점:* **데이터로부터 최적의 노이즈 제거 함수를 학습**하므로 **복잡한 구조와 텍스처도 유지하면서 노이즈 억제**가 가능합니다. 특히 딥러닝으로 **최첨단 PSNR/SSIM 성능**을 달성하여, 가우시안 노이즈 제거의 경우 인간 시각으로 거의 깨끗한 수준까지 복원이 가능합니다. 다양한 네트워크 구조를 통해 **속도-성능 절충**을 조절할 수 있고, 한 번 훈련된 모델은 특정 응용에 빠르게 적용 가능합니다.  
  - *한계:* **충분한 양질의 학습 데이터**가 필요하고, 특정 노이즈 분포에 맞춰진 모델은 **다른 유형의 잡음에 일반화되지 않을 수 있음**이 한계입니다. 또한 **블라인드 노이즈 제거**(노이즈 유형/세기가 변하는 상황)에는 하나의 모델로 완벽 대응이 어렵고, 학습데이터와 도메인이 다르면 성능이 급격히 떨어질 수 있습니다. 아키텍처가 복잡해지면 연산량이 증가하고, 내부 동작이 블랙박스화되어 **해석이 어려운 점**도 있습니다.

- **GAN 및 생성 모델 기반**:  
  - *장점:* **픽셀 단위 오류 최소화 대신 시각적 품질 극대화**를 목표로 하므로, **날카롭고 상세한 복원 결과**를 얻을 수 있습니다. 또한 **짝지은 데이터가 없어도** 훈련할 수 있어 데이터 수집 부담을 줄여줍니다. 노이즈 모델링을 통한 데이터 증강으로 **실제 노이즈에 대한 대응력**을 높이는 등 유연성이 큽니다.  
  - *한계:* **훈련 난이도**가 높고, 모니터링할 지표가 명확하지 않을 수 있습니다 (진짜같은 거짓 세부가 추가되더라도 PSNR은 낮을 수 있음). GAN 특유의 **불안정성**으로 학습 모드 붕괴나 **아티팩트** 발생 위험이 있으며, 결과의 **신뢰성**(진짜 영상과 구분 어려운 인공패턴 생성 가능)에 대한 고려가 필요합니다. 자체 확률적 생성 특성 때문에 **일관성 있게 항상 같은 출력이 보장되지 않을** 수도 있습니다.

- **Transformer 및 Attention 기반**:  
  - *장점:* **전역 컨텍스트**를 활용하여, CNN이 놓칠 수 있는 **멀리 떨어진 픽셀 간 상관관계**까지 포착합니다. 이에 따라 **복잡한 주기적 패턴이나 전역적인 이미지 구조 보존**에 유리하며, 최근 방법들은 노이즈 제거 뿐만 아니라 복원 전반에서 **최고 성능**을 경신하고 있습니다 ([Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf#:~:text=comprehensive%20experiments%20and%20demonstrate%20state,architectural%20designs%20and%20experimental%20choices)). 또한 모듈화된 Transformer 블록은 병렬화에 유리해 **하드웨어 가속** 측면의 장점도 있습니다.  
  - *한계:* 대체로 **매우 많은 학습 데이터**와 **연산 자원**을 필요로 하며, 작은 데이터셋에서는 효과가 제한될 수 있습니다. 또한 **모델 사이즈가 크고 느린 추론 속도**는 실시간/모바일 적용의 걸림돌입니다. 구조적으로 CNN에 비해 **사전 지식(Inductive bias)**이 적어, 복잡한 모델을 작은 데이터에 훈련하면 **과적합 위험**이 있습니다. 따라서 실제 응용에서는 Transformer를 **경량화하거나 CNN과 결합**하는 등의 추가 연구가 필요합니다.

## 6. 결론 및 최신 트렌드 동향

이미지 노이즈 제거 기술은 **간단한 필터링 기법에서 시작하여 딥러닝을 거쳐 Transformer와 자기지도 학습으로 진화**해 왔습니다. **전통 기법**들은 계산 효율성과 해석 용이성 덕분에 현재도 일부 실시간 시스템이나 사전 처리로 활용되고 있으며, **딥러닝 CNN 기법**은 대부분의 일반적인 노이즈 제거 문제에서 사실상의 표준으로 자리잡았습니다. 최근 **최신 트렌드**는 다음과 같이 요약될 수 있습니다:

- **실제 환경 노이즈 처리**: 카메라 센서나 저조도 환경 등에서 발생하는 복잡한 노이즈를 다루기 위해, 실제 데이터를 통한 **모델의 견고성 향상**이 중요해졌습니다. 이를 위해 **노이즈 프로파일링**, **도메인 적응**, **자기지도 학습** 등이 혼합된 기법들이 연구되고 있습니다. 예컨대, **센서별 노이즈 모델을 추정**하여 그 정보를 네트워크에 제공하거나, 합성 데이터로 학습한 모델을 **소량의 실제 데이터로 미세 조정**하는 전략 등이 활용됩니다.

- **평가 지표의 다변화**: 과거에는 PSNR, SSIM과 같은 **수치 지표**가 주로 사용되었으나, 이제는 **LPIPS** 등의 **학습기반 지각 품질 지표**나 실제 인간 판단에 더 근접한 평가가 중시됩니다 ([Heterogeneous window Transformer for image denoising](https://arxiv.org/html/2407.05709v2#:~:text=the%20proposed%20method%20in%20image,For%20instance)). 이는 단순히 노이즈를 제거하는 것에서 나아가 **디테일을 얼마나 잘 보존했는지**, **보기 좋은 이미지인지**까지 고려하는 방향으로 연구가 변모하고 있음을 보여줍니다. 따라서, GAN이나 perceptual loss의 사용도 이러한 추세와 맞물려 계속될 것으로 보입니다.

- **새로운 모델과 융합**: Transformer의 성공 이후에도 완전히 CNN을 대체하기보다는 **각자의 장점을 융합**하는 시도가 늘고 있습니다. 예를 들어 **CNN+Transformer 하이브리드**나, **Diffusion 모델+CNN** 결합 등 여러 형태의 아키텍처 실험이 진행 중입니다. 목표는 **노이즈 제거 성능을 극한까지 높이면서도 효율성과 범용성을 확보**하는 것입니다. 특히 **Diffusion(확산) 모델**은 원래 이미지 생성을 위해 제안되었으나, 그 과정 자체가 **노이즈를 단계적으로 제거**하는 것이기 때문에 노이즈 제거와 근본적으로 연관되어 있습니다. 향후엔 Diffusion 모델을 이용한 **사전학습된 노이즈 제거자(Denoiser) 형태의 사전**이 등장할 가능성도 논의되고 있습니다.

- **어플리케이션 특화 및 통합**: 노이즈 제거는 단독 응용뿐 아니라, **저강도 저품질 영상의 선행처리**로 많이 사용됩니다 (예: 자율주행 비전에서 센서 노이즈 제거, 의료영상 노이즈 제거 등). 최신 연구들은 특정 응용에 맞게 노이즈 제거기를 최적화하거나, **노이즈 제거+디블러링+초해상도** 등을 한 모델에서 공동 처리하려는 시도도 있습니다. 예를 들어 하나의 **멀티태스크 모델**이 다양한 저레벨 복원 작업을 모두 해내도록 하여 실용성을 높이고자 합니다 ([[2012.00364] Pre-Trained Image Processing Transformer](https://arxiv.org/abs/2012.00364#:~:text=task%20%28e.g.%2C%20denoising%2C%20super,level%20benchmarks.%20Code%20is)).

종합하면, **컴퓨터 비전의 노이즈 제거 연구는 꾸준히 발전하며** 현재도 활발히 진행 중입니다. 전통적 기법의 원리들은 딥러닝 모델 속에 녹아들어 **하나의 레이어나 모듈**로 변형되었고 ([Heterogeneous window Transformer for image denoising](https://arxiv.org/html/2407.05709v2#:~:text=Qualitative%20analysis%20is%20used%20to,48%5D%2C%20image)), CNN으로 시작된 딥러닝 흐름은 GAN과 자기지도 학습을 거쳐 Transformer로 확장되었습니다. 최신 방법들은 **노이즈 제거 성능뿐만 아니라 데이터 조건, 실행 속도, 시각적 품질 등 다양한 측면에서 균형 잡힌 발전**을 추구하고 있습니다. 앞으로는 **대규모 사전학습된 복원 모델**의 활용과, **다양한 노이즈 유형에 대한 범용적인 복원 기법**이 주요 이슈가 될 것으로 전망됩니다. 이러한 흐름을 바탕으로, 노이즈 제거 기술은 더욱 정교해져서 깨끗하고 품질 높은 영상 획득에 기여할 것입니다.

**참고문헌:** 주요 내용은 Buades et al. (2005) ([](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/Buades-NonLocal.pdf#:~:text=the%20denoised%20value%20at%20x,2)), Dabov et al. (2007) ([An Analysis and Implementation of the BM3D Image Denoising Method](https://www.ipol.im/pub/art/2012/l-bm3d/article_lr.pdf#:~:text=BM3D%20is%20a%20recent%20denoising,We%20hope%20this%20new)) 등의 **기초 연구**, Burger et al. (2012) ([](https://people.tuebingen.mpg.de/burger/neural_denoising/files/neural_denoising.pdf#:~:text=and%20noise,a%20patch%20con%02tains%20enough)), Zhang et al. (2017) ([[1608.03981] Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://arxiv.org/abs/1608.03981#:~:text=attracting%20considerable%20attentions%20due%20to,is%20able%20to%20handle%20Gaussian)) 등의 **딥러닝 선구적 연구**, Lehtinen et al. (2018) ([[1803.04189] Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/abs/1803.04189#:~:text=,based%20on%20noisy%20data%20only))의 **자기지도 학습**, Chen et al. (2021) ([[2012.00364] Pre-Trained Image Processing Transformer](https://arxiv.org/abs/2012.00364#:~:text=is%20trained%20on%20these%20images,URL%20and%20this%20https%20URL)), Zamir et al. (2022) ([Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf#:~:text=comprehensive%20experiments%20and%20demonstrate%20state,architectural%20designs%20and%20experimental%20choices)) 등의 **최신 Transformer 기반 연구를 포함한 문헌**을 기반으로 작성되었습니다.



---

# 참고문헌
[^1]: Lin, M., Chen, Q., & Yan, S. (2013). *Network in network*. **arXiv preprint arXiv:1312.4400**.

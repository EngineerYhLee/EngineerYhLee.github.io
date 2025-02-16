---
published: false
title: "역전파 기법(Backpropagation)"
description: "역전파 기법의 이해와 구현"
header:
  teaser: /assets/images/backpropagation.png
  og_image: /assets/images/backpropagation.png
  image_description: "Backpropagation"
date: 2025-02-16
last_modified_at: 2025-02-16 10:00:00
toc: true
toc_sticky: true
use_math: true
categories:
  - 
tags:
  - 
---

[//]: # (작성 양식)
[//]: # (# 마크다운: kramdown)
[//]: # (# 수식 예시 => 수식이 필요할 경우에만 작성:  $$y = f&#40;z&#41; = f\left&#40;\sum_{i=1}^n w_i x_i + b\right&#41;$$)
[//]: # (# 언어: 한글, 주요 단어는 영문과 같이 표시)
[//]: # (# 참고문헌: 실제 존재하는 논문들을 형식에 맞게 기재)

## 1. 인용 예시 
- **LeNet-5 등장**  
  얀 르쿤과 동료들은 LeNet-5를 비롯한 초기 CNN 모델을 개발하여, 손글씨 숫자 인식(MNIST) 등 실용적 문제에서 탁월한 성능을 입증했습니다[^1].  

# 참고문헌

[^1]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.  
[^2]: Hubel, D. H., & Wiesel, T. N. (1959). Receptive fields of single neurones in the cat's striate cortex. *The Journal of Physiology*, 148(3), 574-591.  
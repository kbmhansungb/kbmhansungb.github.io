# GAN
Generative Adversarial Networks (GANs)은 심층 신경망 설계에서 아키텍처 변화를 대표합니다. 이 아키텍처는 제한된 데이터로 일반화할 수 있으며, 소규모 데이터셋에서 새로운 장면을 창조하고, 시뮬레이션된 데이터를 더 현실적으로 보이게 할 수 있는 여러 가지 이점을 가지고 있습니다. 많은 오늘날의 기술들은 대량의 데이터를 요구하는데, 이 새로운 아키텍처를 사용함으로써 필요한 데이터의 양을 대폭 줄일 수 있습니다. 극단적인 예에서는 이러한 유형의 아키텍처가 다른 유형의 딥러닝 문제에 필요한 데이터의 10%만을 사용할 수도 있습니다.

### GAN 구축을 위한 비유
전형적인 비유는 위조범(Generator)과 FBI 요원(Discriminator)입니다. 위조범은 FBI 요원의 검사를 통과할 수 있는 새로운 방법을 지속적으로 모색하고 있습니다.

### GAN 구현의 작동 방식
게임 이론의 문제인 미니맥스 문제로 구조화된 이 방식은 생성자와 구분자가 각각의 목표를 가지고 경쟁하는 구조입니다.

### GAN 프레임워크
GAN 프레임워크에서 생성자는 구분자와 함께 훈련을 시작하며, 대립적인 훈련을 시작하기 전에 구분자는 이미지를 분류할 수 있을 정도로 몇 에포크 동안 훈련해야 합니다. 이 구조의 마지막 부분은 손실 함수입니다. 손실 함수는 생성자와 구분자의 훈련 과정을 멈출 기준을 제공합니다.

### 기본 빌딩 블록 - 생성자
생성자는 가장 중요한 구성 요소 중 하나입니다. 궁극적으로 생성자는 전체 훈련 과정이 끝난 후 우리가 보게 될 이미지나 출력물을 생산합니다. 생성자는 적대적 모드에서 훈련되며, 이는 생성자와 구분자를 모델에 함께 연결하는 과정을 포함합니다.

### 기본 빌딩 블록 - 구분자
생성자가 데이터를 생성하는 반면, 구분자는 생성된 이미지와 실제 이미지를 판별하는 데 사용됩니다. 구분자는 일반적으로 간단한 CNN(Convolution Neural Network)입니다.

## 훈련 및 추론
모델을 훈련하는 방법과 훈련된 모델에서 데이터를 얻는 방법에 대해 간략히 다루는 섹션도 있습니다.

## 손실 함수
손실 함수는 GAN의 핵심입니다. 이는 생성자의 결과물이 실제 이미지와 구분할 수 없을 정도로 발전하였을 때, 학습을 중단하는 기준점을 제공합니다.

## 참고
- [kaggle, "Generative Dog Images"](https://www.kaggle.com/competitions/generative-dog-images/discussion/99215)
- [Medium, "Inside the Generative Adversarial Networks (GAN) architecture"](https://medium.com/@Packt_Pub/inside-the-generative-adversarial-networks-gan-architecture-2435afbd6b3b#:~:text=Generative%20Adversarial%20Networks%20%28GANs%29%2C%20represent%20a%20shift,neural%20networks.)

Z-Fighting은 두 개 이상의 개체가 동일한 깊이에서 렌더링될 때 발생하는 그래픽 문제로, 개체가 깜박이거나 우선순위를 놓고 다투게 합니다. 이는 개체가 서로 너무 가깝거나 카메라의 근거리 및 원거리 클리핑 평면이 서로 너무 가깝게 설정된 경우에 발생할 수 있습니다.

Z-Fighting을 피하기 위해서는

1. 물체나 표면이 겹치는 것을 최대한 피하십시오.
2. 고해상도 텍스처와 모델을 사용하여 겹치는 표면의 필요성을 줄입니다.
3. 렌더링에 사용되는 깊이 버퍼의 정밀도를 높입니다.
4. 오클루전 컬링 기술을 사용하여 렌더링해야 하는 개체 수를 줄입니다.
5. 정확한 깊이 정보의 필요성을 줄이는 물리적 기반 렌더링 기술을 사용합니다.
6. Z-파이팅의 가능성을 높일 수 있는 동일한 깊이 또는 그 근처에 표면을 배치하지 마십시오.
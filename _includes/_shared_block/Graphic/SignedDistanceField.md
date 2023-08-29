[SDF 합치기 정리](https://joyrok.com/SDFs-Part-Two)

**SDF는 "Signed Distance Field"의 약자입니다. 공간의 임의 지점에서 객체 또는 표면의 가장 가까운 경계까지의 거리를 정의하는 스칼라 필드입니다.** 거리 값의 부호는 점이 객체의 내부 또는 외부에 있는지를 나타내며 양수 값은 외부를 의미하고 음수 값은 내부를 의미합니다. SDF는 메시 또는 암시적 함수로 정의된 것과 같은 보다 복잡한 모양뿐만 아니라 구 및 상자와 같은 단순한 기하학적 모양을 나타내는 데 사용할 수 있습니다.

컴퓨터 그래픽에서 SDF는 일반적으로 레이 마칭(ray marching) 및 구조적 솔리드 지오메트리와 같은 기술에 사용되며, 여기서 객체 교차를 테스트하고 음영을 계산하는 빠르고 효율적인 방법을 제공합니다. 객체를 SDF로 나타내면 복잡한 기하학적 데이터가 아닌 간단한 스칼라 값에 대해 이러한 작업을 수행할 수 있으므로 매우 상세하고 복잡한 장면을 실시간으로 렌더링할 수 있습니다.

```cpp
float circleSDF(float x, float y, float cx, float cy, float r) {
  float dx = x - cx;
  float dy = y - cy;
  return std::sqrt(dx * dx + dy * dy) - r;
}
```

이 함수는 중심 과 반지름 (x, y)이 있는 원의 경계에서 점에서 가장 가까운 점 까지 부호 있는 거리를 계산합니다 . 결과는 점에서 원의 중심까지의 거리와 원의 반지름 간의 차이입니다. 결과가 음수이면 점이 원 안에 있습니다. 양수이면 포인트가 외부에 있습니다.

이것은 하나의 예일 뿐이지만 유사한 SDF 함수를 상자, 구 및 암시적 함수로 정의된 훨씬 더 복잡한 모양과 같은 다른 모양에 대해 작성할 수 있습니다. 코드에서 SDF를 사용하면 객체 교차 테스트 및 음영 계산을 간단하고 빠르고 유연한 방식으로 수행할 수 있으므로 컴퓨터 그래픽 및 실시간 렌더링에서 강력한 도구가 됩니다.
---
layout: post
title: Lighting
tags : [Graphics, Light]
---

- [ ] 언리얼 라이팅 어렵지 않아요
- [ ] [Naughty Dog 수석 아티스트의 UE4 조명 팁](https://www.reddit.com/r/unrealengine/duplicates/arudy2/ue4_lighting_tips_from_senior_naughtydog_artist/)

## 언리얼 라이팅

[인바이런먼트 라이팅](https://docs.unrealengine.com/5.0/ko/lighting-the-environment-in-unreal-engine/)

버추얼 월드를 구축할 때 중요한 부분 중 하나는 적절한 라이팅 방법을 택하는 것입니다. 이는 작은 밀폐 공간 씬을 작은 라이트로 효과적으로 비추는 것부터 커다란 월드를 하나의 지배적인 라이트 소스로 비추는 것까지 모든 라이팅 방법을 의미합니다. 엔진에서는 프로젝트의 요구 결과를 달성하는 데 필요한 라이팅 옵션과 툴을 제공합니다.

이 페이지의 주제는 엔진에서 제공하는 다양한 라이팅 피처와 툴에 대한 정보이며, 언리얼 엔진에서 씬을 라이팅하기 위한 세부적인 학습과정을 모두 안내하는 가이드도 포함되어 있습니다.

* 라이팅은 빛을 사용해 물체를 보는 것 뿐만 아니라, 분위기, 감정, 시간대별 등 연출하는 용도로 사용된다고 합니다. 
* **이를 전문적으로 하는 사람들을 라이팅 아티스트라고 합니다.**

<details><summary>라이트 유형 및 모빌리티</summary>
<div markdown="1">

[라이트 유형 및 모빌리티](https://docs.unrealengine.com/5.0/ko/light-types-and-their-mobility-in-unreal-engine/)

언리얼 엔진은 거의 모든 유형의 라이팅 시나리오를 생성하고 소규모 월드부터 대규모 월드까지 작업할 수 있는 여러 라이트 유형을 제공합니다. 이러한 각 라이트 유형에는 저마다의 모빌리티 옵션이 있고, 이 옵션으로 라이트가 씬 내의 다른 액터와 상호작용하는 방식, 라이팅 시스템이 라이트를 활용하는 방식을 정의합니다.

언리얼 엔진은 다음과 같은 유형의 라이트를 제공합니다.

* 디렉셔널 라이트(Directional Lights)는 주요 야외 라이트 또는 굉장히 멀거나 거의 무한히 떨어진 거리에서 드리워지는 것으로 보이는 라이트입니다.
* 스카이 라이트(Sky Light)는 씬의 배경을 캡처하여 레벨의 지오메트리에 적용합니다.
* 포인트 라이트(Point Lights)는 전구처럼 작동하며 단일 포인트에서 모든 방향으로 라이트를 드리웁니다.
* 스포트 라이트(Spot Lights)는 단일 포인트에서 원뿔 세트에 의해 제한된 방향으로 라이트를 발산합니다.
* 렉트 라이트(Rect Lights)는 사각형 표면에서 한 방향으로 라이트를 발산합니다.

각각의 라이트는 다음과 같이 사용합니다.

* Directional 및 sky light는 대규모 익스테리어에 유용, 인테리어의 틈새를 통한 라이팅 및 새도잉을 제공할 때도 좋음. 대규모 익스테리어에서 디렉셔널 라이트는 밀집한 폴리지 및 기타 지오메트리에 가장 효율적으로 라이팅 구현.
* 포인트, 스포트, 렉트 라이트는 보다 소규모의 국소적 영역에 라이팅을 구현하는 데 유용, 라이트 유형 및 프로퍼티는 주어진 씬 내의 라이트 셰이프와 룩을 정의하는 데 유용, 또한 이러한 라이트 유형은 동일한 프로퍼티를 다수 공유.

<br>

**씬 내의 각 액터 유형에는 게임플레이 도중 어느 정도 이동 또는 변경이 가능한지 여부를 제어하는 모빌리티(Mobility) 세팅이 있습니다.** 라이트 액터의 경우, 모빌리티 선택으로 라이트 빌드 시 씬에서 라이트가 어떻게 처리되는지를 정의합니다.

* 스태틱(Static) 모빌리티는 게임플레이 도중 어떤 식으로든 이동 또는 업데이트되지 않는 라이트 액터에 사용됩니다.
* 스테이셔너리(Stationary) 모빌리티는 게임플레이 도중 변경될 수는 있지만 이동되지는 않는 액터에 사용됩니다. 
* 무버블(Movable) 모빌리티는 게임플레이 도중 추가되거나, 제거되거나, 이동되어야 하는 라이트 액터에 사용됩니다.

각각의 모빌리티는 다음과 같이 사용합니다.

* 스태틱 라이트는 라이트매스를 사용하여 사전 계산된 라이트맵에 기여합니다. 이러한 라이트는 씬을 조명하고 스태틱 및 스테이셔너리로 설정된 액터에 대한 라이팅 데이터를 생성합니다. 하지만 무버블 액터의 경우 간접 라이팅 캐시 또는 볼류메트릭 라이트맵에 저장된 라이팅 데이터로 라이팅이 구현됩니다.
* 스테이셔너리 라이트는 게임플레이 도중 색 또는 강도 등이 변경되거나 완전히 꺼지는 등, 어느 정도 변경될 수 있습니다. 스테이셔너리 라이트는 라이트매스를 사용하여 사전 계산된 라이트맵에 기여하지만 무버블 오브젝트에 대해서도 다이내믹 섀도를 드리울 수도 있습니다. 스테이셔너리 라이트는 추가 비용이 동반되며, 언제든 단일 오브젝트에 영향을 미칠 수 있는 라이트 수가 제한되어 있습니다. 예를 들어 단일 오브젝트는 언제든 최대 4개의 스테이셔너리 라이트에 의해서만 영향을 받을 수 있습니다.
* 무버블 라이트는 다이내믹 섀도만 드리웁니다. 게임플레이 도중 이동할 수 있을 뿐 아니라 색, 강도, 기타 라이트 프로퍼티를 필요에 따라 변경할 수 있습니다.
* 무버블 모빌리티는 섀도잉 비용이 가장 높기 때문에 사용 시에 주의가 필요합니다. 그러나 섀도잉이 없는 무버블 라이트는 계산 비용이 굉장히 낮으며, 디스크에 저장해야 할 라이팅 데이터가 없기 때문에 스태틱으로 설정된 라이트보다 비용이 낮습니다.
* 라이트 모빌리티 유형은 퍼포먼스, 룩 및 디자인 선택에 영향을 미치므로 프로젝트에 가장 잘 맞는 라이트 모빌리티 유형을 선택하는 것이 중요합니다. 일부 라이트 기능은 플랫폼 또는 머신에 따라 기능이 제한되거나 부분적으로 지원되지 않을 수도 있습니다. 예를 들어 다이내믹 섀도잉은 모바일 플랫폼의 모든 라이트 유형에서 지원되지 않습니다.

</div></details>

<details><summary>모범 사례 : 언리얼 라이팅 어렵지 않아요</summary>
<div markdown="1">

[언리얼 라이팅 어렵지 않아요](https://www.youtube.com/watch?v=1UUG_CHaBWk)

**`라이팅 어렵지 않아요` 했지만, 라이팅은 어렵습니다.**  기술적인 장벽은 되게 낮지만, 언리얼의 라이팅을 잘 활용하는 것은 어렵습니다. 개인적인 경험을 토대로 작성했기 때문에, 절대적인 진리는 아닙니다.
</div></details>

<details><summary>모범 사례 : `햇살 좋은 날`라이팅 구현 팁</summary>
<div markdown="1">

['햇살 좋은 날' 라이팅 구현 팁](https://www.youtube.com/watch?v=5Eh3o3Op81w)

작년 언리얼 보자 우리 아이 를 출품한 햇살 좋은 날에 제작하기 전 라이트 베이 그 관련 자료를 찾아보고 연구 했었는데 수많은 라이트 옵션과 각자 다른 애셋을 사용한 여러 가지 환경 셋팅 들이 있었습니다 결론은 라이팅에 정답은 없고 구현하고자 하는 방향대로 환경 값들을 세밀하게 조절해 나가야 합니다 라이팅을 구현하는 여러 가지 방법 중 하나로 이 영상을 참고해 주셨으면 합니다.

1. 씬 구성 및 환경 세팅
2. 라이트 메시와 라이트 세팅
3. 디테일 작업 및 마무리 작업


라이트에 사용한 주 재료는 DirectionalLight, SkyLight, ExponentialHeightFog를 사용했습니다.

주광은 DirectionalLight를 환경관은 SkyLight, ExponentailHeightFog를 사용했습니다.

라이트의 모빌리티는 구워진 라이트 맵과 스테이셔너리 라이트 맵을 사용했습니다.

* 라이트 설정하는데 방해 될 수 있으므로 AutoExposure를 해제합니다. 포스트 프로세서 불륨 안에 있는 Exposure탭으로 가서, 3으로 설정하고, Min과 Max값을 모드 1로 설정합니다. 이렇게 하면 화면이 밝아지거나, 어두워지지 않아서 작업하기 편해집니다.

라이트 환경 세팅을 해 보겠습니다. 라이트가 잘 구워질 수 있는 환경을 만듭니다.

라이트 메스 관련 액터와 라이트 메스 세팅 후, 디렉셔널 라이트가 따듯한 색을 내도록 만들고, Exponentail Height Fog로 환경색을 잡습니다.

* 창문에 라이트 메스 포탈을 설치합니다. 라이트 메스 포탈을 설치하면, 더많은 빛이 창문으로 들어오므로, 더 많은 빛이 들어오는 느낌을 낼 수 있습니다.

* 라이트 메시 볼륨을 설치하여 볼륨 안에서, 라이트 메스 계산을 집중시킵니다. 그다음 월드 세팅으로 가서, 디렉셔널 라이트 반사 횟수를 20정도로 조절, 스카이 라이트 반사 횟수도 20정도로 조절합니다.

무버블로 설정했던, 메시를 이제 스테틱으로 변경하고, 라이트를 굽습니다. 라이팅 빌드가 완료되는 환해집니다.

* 그림자의 상태가 좋지 않으므로, 케스케도우 쉐도우 맵을 활성화 시킵니다. 케스케이드 값을 변경하면, 그림자가 선명해 집니다.

이제 방안에 따듯한 느낌이 나도록, 디렉셔널 라이트를 수정합니다. 그리고 다시 라이트를 빌드합니다.

* 색온도 값을 참고하여, 색온도를 사용하면, 따듯하고 차가운 느낌을 내기 쉽습니다.

원하는 하늘 생삭으로, 스카이 값을 변경합니다. 그리고 방안에 반사값을 가진 물체가 더 빛을 잘 반사할 수 있도록, 스피어 리플렉션을 배치합니다. 그리고 라이트를 다시 굽습니다.

* 라이트 퀄리티를 높이기 위해서, 라이트 품질을 높일 수 있습니다.

방 모서리에 빛이 새어들어옵니다. 최적화 뷰 모드에서 라이트맵 밀도를 설정할 수 있습니다. 라이트 탭안에 오버라이드된 라이트 리졸루션값을 조절하고, 다시 라이트를 빌드합니다.

햇살이 들어오는 느낌이 들도록, Fog를 수정합니다. Volumetric Fog를 활성화 하고, Extention Scale 값을 10으로 수정하고, 스케터링 디스트리 뷰션 값을 0 이하로 설정합니다. 햇살이 보이지만 잘 보이지 않으므로, 빛의 세기를 조금더 강하게 합니다.

포인트 라이트를 무버블로 변경하고, 라이트를 베이크 하지 않는 방식으로 진행합니다. 또한 그림자도 사용하지 않고, 색감만 더하는 방식으로 사용합니다. 이렇게 색감을 배치합니다.

색들이 좀더 화려하게 보일 수 있도록 포스트 프로세스 값을 조정합니다. 블룸에, 인텐시티 값을 올리고, 스레시 홀드 값을 조정해서 사용합니다.

그리고 컬러 그레이딩 값을 조정합니다. 채도가 좀더 높아보이게 조정할 수 있습니다. 컨트라스트 값도 조정해서, 색 대비도 조절합니다.

대비가 강해져서 어두워지는 부분들이 보이는데, 이 부분은 감마값과 게인 값을 올려서 조절합니다.

이제 시네마틱 카메라 액터를 설치합니다. 시네마틱 카메라 액터를 설치하고, 초점을 맞춥니다.

DOF가 강하게 되어있어서, 약간 조절합니다.

화면을 좀더 선명하게 보고 싶다면, 콘솔 명령어 r.톤매퍼.샤픈 명령어를 적용합니다. 이러면 보다 선명한 값을 느낄 수 있습니다.

</div></details>

## 조명

[Lighting](https://en.wikipedia.org/wiki/Lighting)

[Three-point lighting](https://en.wikipedia.org/wiki/Three-point_lighting)

### 3점 조명

주광(Key light)은 물체의 앞쪽에서 직접 비추는 라이트입니다. **주광의 세기나 색깔, 각도 등은 물체 조명의 전반적인 모습을 결정합니다.** **실내 조명에서, 주광은 주로 특별한 전등이나, 카메라 플래시입니다. 하지만 낮의 실외 조명에서는, 태양이 주광의 역활을 합니다.**

보조광(Fill light, 채우는 빛)는 물체를 비추지만, 주광가는 달리 측면에서 비춥니다. 이는 주광에 의해 그늘진 부분을 비추어, 명암의 대조를 줄이거나 없애는 효과를 줍니다. 보조광은 주로 주광보다는 덜 밝고, 부드러운 조명을 씁니다. **보조광을 쓰지 않으면, 그림자 때문에 적나라한 대조가 나타납니다. 때로는 일부러 약한 주광 조명을 쓰기도 합니다.**

후광(Back light)는 물체를 뒤에서 비춥니다. **이는 물체의 가장자리에 빛을 주거나, 물체를 배경과 분리해 윤곽선을 강조하는데 사용됩니다.**

### 4점 조명
배경 조명(Background light)는 물체의 뒤에 설치됩니다. 전경을 비추는 세 조명과는 다르게, 이는 벽이나 바깥 풍경과 같은 배경을 비춥니다. **이 기술은 배경에 생기는 전경의 그림자를 없애거나, 배경에 더 많은 주의를 끌기 위해서, 카메라로 찍은 사진의 깊이감을 갖게 하기 위해서 쓰입니다.**

[오클루전 컬링](https://docs.unity3d.com/kr/2020.3/Manual/OcclusionCulling.html)

오클루전 컬링은 Unity가 다른 게임 오브젝트에 의해 뷰에서 완전히 가려진(오클루전된) 게임 오브젝트에 대한 렌더링 계산을 수행하지 못하도록 하는 프로세스입니다.

프레임마다 카메라는 씬의 렌더러를 검사한 후 그릴 필요가 없는 렌더러를 제외(컬링)합니다. 기본적으로 카메라는 절두체 컬링을 수행하여 카메라의 뷰 절두체에 속하지 않는 모든 렌더러를 제외합니다. 하지만 절두체 컬링은 렌더러가 다른 게임 오브젝트에 의해 가려지는지를 확인하지 않으므로, Unity가 최종 프레임에 표시되지 않는 렌더러에 대한 렌더링 작업에 CPU 및 GPU 시간을 여전히 낭비할 수 있습니다. 오클루전 컬링은 Unity가 이러한 낭비되는 작업을 수행하지 못하도록 방지합니다.

유니티에서 오클루전 컬링을 사용하려면 다음과 같은 단계를 따르면 됩니다.

1. 상단 메뉴에서 Window > Rendering > Occlusion Culling 을 선택하여 오클루전 컬링 창 을 엽니다.
2. Bake 탭을 선택합니다.
3. 인스펙터 창의 오른쪽 하단 모서리에 있는 Bake 버튼을 누릅니다. Unity가 오클루전 컬링 데이터를 생성하고, 해당 데이터를 프로젝트에 에셋으로 저장하고, 해당 에셋을 현재 씬과 연결합니다.
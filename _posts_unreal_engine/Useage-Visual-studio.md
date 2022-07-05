---
layout: post
title: Visual studio setting
---

* 비쥬을 스튜디오 프로젝트를 새로 생성할 때 마다 설정했던 내용이 초기화됩니다.

* NMAKE는 Microsoft의 make 도구 구현입니다. Microsoft프로그램 유지 관리 유틸리티(NMAKE.EXE)는 설명 파일에 포함된 명령을 기반으로 프로젝트를 빌드하는 32비트 도구입니다.

* [언리얼 엔진용 Visual Studio 구성](https://docs.unrealengine.com/4.27/ko/ProductionPipelines/DevelopmentSetup/VisualStudioSetup/)

언리얼 엔진의 기본적인 세팅은 공식문서에 잘 설명되어 있습니다.

* [Visual studio 2019를 사용할 때 IntelliSense를 빠르게 하는 방법](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=ratoa&logNo=221950619537)   

1. VC++디렉터리의 포함 디렉터리의 내용을 NMake의 IntelliSense탭에 있는 포함 검색 경로에 복사해 붙여넣습니다.
2. NMAKE의 IntelliSense탭에 있는 추가 옵션에 [/Yu](https://docs.microsoft.com/ko-kr/cpp/build/reference/yu-use-precompiled-header-file?view=msvc-170) 옵션을 추가합니다.

원본 파일을 수정할 때마다 IntelliSense는 제공된 모든 경로를 검색하여 원본에서 만들어진 참조를 찾으려 하지만, 언리얼 엔진 프로젝트는 매우 크기가 크기때문에 모든 것들을 통과하는 데는 많은 시간이 걸립니다. 그러나 라이브러리 코드는 전혀 변경되지 않으므로 텍스트 편집기에서 검색할 필요가 없습니다. 모든 안정적인 코드가 포함된 미리 컴파일된 헤더 파일을 만들 수 있고 참조로 사용할 수 있도록 IntelliSense를 사용할 수 있습니다.

/Yu를 추가 옶션에 넣음으로써 IntelliSense에 미리 작성된 헤더 파일을 사용하도록 지시합니다.
(include했는데 식별자가 정의되지 않았다고 나오는 문제가 생길 수 있습니다.)

* [Visual studio에서 인텔리센스가 작동하지 않을때](https://forums.unrealengine.com/t/intellisense-stopped-working/384124/6)

1. 솔루션을 닫고 솔루션을 다시 만듭니다.
2. 인텔리센스를 다시 스캔합니다.

* [<> "" 차이](https://kinotion.tistory.com/453)

```cpp
#include "temp.h"
#include <temp.h>
```

""은 개발자가 구체적으로 지정해 놓은 결로에서 헤더 파일을 찾고, <>은 대게 시스템에서 가지고 있는 헤더파일을 include하는데 사용합니다.

따라서 ""은 구체적인 경로가 들어갈 경우도 있지만<>은 이미 경로를 컴파일러가 감지하고 있기 때문에 구체적 경로가 적히지 않는 것입니다.

* 토마토 설치
토마토를 설치하니 인텔리센스가 보이지 않습니다. 제가 못하는 거일 수도 있지만, 뭐 아무튼, 오류가 안보이므로 정적분석이 필요합니다. 정확한 수는 모르지만 인텔리센스 오류가 만 몇천개 넘어가면 작동하지 않습니다.

## 정적분석을 위한 PVS Studio
일자 : 2022 05 06
IDE : Visual studio community 2022
환경 : Unreal engine 5.0.1

* 설치방법
    1. Visual studio 확장에서 PVS studio를 찾아서 다운로드합니다.
    2. Visual studio를 끄고, 설치를 진행합니다. 
        - 또한 기본경로로 설치하는 것이 좋습니다. 언리얼에서 ThirdParty경로 또는 C의 경로에서 찾습니다.
    3. [Unreal Build Tool 통합을 사용한 분석](https://pvs-studio.com/en/docs/manual/0043/)을 따라 설치합니다.
* " -StaticAnalyzer=PVSStudio " 입니다.
* PVS가 작동되는 것을 확인합니다.
    - 이제 작동된다면 정적분석을 이용해서 오류를 어떻게 잡을 수 있을지 고민해봐야 합니다.

<details>
<summary>삽질 기록</summary>

[작동이 안되서 포럼에다 올려봤습니다.](https://forums.unrealengine.com/t/unable-to-find-pvs-studio/550089)

버그라고 합니다. 다음에는 깃헙 이슈목록도 보도록 합니다.

[Unreal Build target이 없다고 합니다.](https://forums.unrealengine.com/t/unrealheadertool-target-deleted-on-project-rebuild/522872/6)

버그라고 합니다. 핫 픽스 나왔으니 버전 업하라고 합니다.

[이번엔 용량이 문제입니다.](https://hkebi.tistory.com/1591)

이참에 관리하기 쉽게 프로그램 하나를 설치하도록 합니다.

</details>


참고자료
: [C++ PVS-Studio 설치 및 사용](https://jacking75.github.io/Cpp_pvs_studio/)
, [Cppcheck and PVS-Studio compared](https://pvs-studio.com/en/blog/posts/0149/)
, [언리얼 엔진 개발 프로세스에 정적 분석 사용하기](https://www.unrealengine.com/ko/tech-blog/static-analysis-as-part-of-the-process)


## 버전업하고 빌드가 안되는 경우

* 첫번째 빌드에서 실패하고 두번째 빌드에서 되는경우, 알수가 없습니다.
    - 로그를 봅니다.
        - 블루프린트가 깨져서 안될 수도 있습니다.

## 셰이더 컴파일을 위한 세팅
셰이더 컴파일을 위해서는 엔진의 Platform.ush를 필요로 합니다. 추가하지 않을시에 "/Engine/Shaders/Public/Platform.ush"를 추가하라고 합니다. 하지만 이는 include되지 않는 문제가 있습니다.

"D:/Program Files/Epic Games/UE_5.0/Engine/Shaders/Public/Platform.ush"로 직접경로로 설정하면 가상경로를 이용하라는 오류를 발생시킵니다.

## 다시빌드 하는것은 시간을 너무 많이 잡아먹습니다.
라이브 코딩을 선호하도록 합시다.
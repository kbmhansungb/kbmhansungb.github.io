---
layout: post
title: Automation test
---

- [ ] 어떻게 테스트코드를 먼저 작성하는가?
- [ ] 어떻게 자체 검증?
- [ ] 기능 테스트 (Functional test)는 클라이언트가 제공한 기능명세를 이용하여 수행하며 소프트웨어의 기능 요구사항을 확인합니다.

## TDD (Test-Driven Development, 테스트 주도 개발)
TDD란 Test Driven Development의 약자로 '테스트 주도 개발'이라고 한다. 반복 테스트를 이용한 소프트웨어 방법론으로, 작은 단위의 테스트 케이스를 작성하고 이를 통과하는 코드를 추가하는 단계를 반복하여 구현합니다. 

짧은 개발 주기의 반복에 의존하는 개발 프로세스이며 애자일 방법론 중 하나인 eXtream Programming (XP)의 'Test-First' 개념에 기반을 둔 단순한 설계를 중요시합니다.


**TDD를 이용하여 다음을 기대할 수 있습니다.**
- 코드가 프로그래머의 손을 벗어나기 전에 빠르게 피드백을 받을 수 있습니다.
- 작성한 코드의 불안정성을 개선하여 생산성을 높일 수 있습니다.
- 프로그래머의 오버 코딩을 방지합니다.
- 테스트 코드를 작성하는 과정에서 히스토리가 남아, 과거 의사결정을 쉽게 상기할 수 있습니다.

<details><summary>추가적으로</summary>
<div markdown="1">

<center>

<div class="mermaid"> 
graph LR;
A(디자인)-->B; 
B(테스트 코드 작성)-->C; B-->A;
C(코드개발)-->D; 
D(리팩토링)-->B; 
</div>

</center> 

이 기법을 개발했거나 '재발견' 한 것으로 인정되는 Kent Beck은 2003년에 TDD가 단순한 설계를 장려하고 자신감을 불어넣어 준다고 말하였다. 

* eXtream Programming(XP)는 미래에 대한 예측을 최대한 하지 않고, 지속적으로 프로토타입을 완성하는 애자일 방법론 중 하나입니다. 이 방법론은 추가 요구사항이 생기더라도, 실시간으로 반영할 수 있습니다.
* 단위 테스트(Unit test)는 말 그대로 한 단위만을 테스트하는 것입니다.

</div></details>

<details><summary>유닛 테스트 (Unit test)</summary>
<div markdown="1">

단위 테스트는 응용 프로그램에서 테스트 가능한 가장 작은 소프트웨어를 실행하여 예상대로 동작하는지 확인하는 테스트입니다.

**단위 테스트에서 테스트 대상 단위의 크기는 엄격하게 정해져 있지 않다. 하지만, 일반적으로 클래스 또는 메소드 수준으로 정해진다. 단위의 크기가 작을수록 단위의 복잡성이 낮아집니다.** 따라서, 단위 테스트를 활용하여 동작을 표현하기 더 쉬워진다. 즉, 테스트 대상 단위의 크기를 작게 설정해서 단위 테스트를 최대한 간단하고 디버깅하기 쉽게 작성해야 합니다.

단위 테스트는 실제 코드를 작성하기 전에 작성해야됩니다. 이 규칙은 TDD를 수행하는 경우 반드시 따라야 하는 규칙입니다.

</div></details>

<details><summary>유닛테스트 작성을 위한 원칙 FIRST</summary>
<div markdown="1">

**FAST**
> **테스트는 빨라야 한다. 여기서 빠름의 기준은 밀리 초(ms)입니다.** 단위 테스트를 테스트하는 데 있어 실행 시간이 0.5 초 또는 0.25 초가 걸리는 테스트는 빠른 테스트가 아닙니다.

하나의 프로젝트에서 적게는 몇백 개에서 많게는 수천 개의 테스트를 할 수 있으므로 테스트의 실행 시간은 빨라야 합니다. 만약 테스트가 느리다면 개발자는 테스트를 주저하게 되고 자주 검증하지 않은 소스코드는 그만큼 버그가 발생할 확률이 높아집니다.

**Independent** 
> **테스트에 사용된 데이터들은 서로 의존하면 안 됩니다.** 테스트에 필요한 데이터는 테스트 내부에서 독립적으로 사용해야 합니다.  

만약 데이터가 서로에게 의존하면 테스트 하나가 실패할 때 나머지도 잇달아 실패하므로 원인을 진단하기 어려워지기 때문입니다.

때론 데이터의 존재 여부를 찾는 테스트가 있는 경우엔 해당 데이터는 테스트 내부에서 생성되어야 하며 나중에 테스트에 영향을 미치지 않도록 제거해야 한다.

**Repeatable**
> **테스트는 어느 환경에서든 반복적으로 테스트를 실행할 수 있어야 합니다.** 여기서 환경은 네트워크 나 데이터베이스에 의존하지 않는 환경을 뜻합니다. 결론적으로 인터넷이 되든 안 되든 데이터베이스에 접속하든 안 하든 언제 어디서나 테스트를 할 수 있어야 합니다.

환경에 의존하지 않는 테스트가 실패할 수 있는 유일한 이유는 오로지 테스트할 클래스 또는 메소드가 제대로 작동하지 않기 때문입니다.

**Selef-Validating**
> **테스트는 자체 검증이 되어야 합니다.** 테스트의 검증은 수작업이 아닌 자동화가 되어야 하는데 테스트가 실행될 때마다 메서드 출력이 올바른지를 확인하는 것은 개발자가 결정해서는 안 됩니다.

**Timely**
> **좋은 단위 테스트는 미루지 않고 즉시 작성**합니다. 단위 테스트는 소프트웨어 개발의 완성도, 품질을 높이는 좋은 습관입니다.

만약 테스트를 제때 작성하지 않고 미루어 작성하지 않는다면 코드에 결함이 발생할 확률이 높아집니다.

</div></details>

<details><summary>통합 테스트 (Integration test)</summary>
<div markdown="1">

**통합 테스트는 단위 테스트보다 더 큰 동작을 달성하기 위해 여러 모듈들을 모아 이들이 의도대로 협력하는지 확인하는 테스트입니다.**

통합 테스트는 단위 테스트와 달리 개발자가 변경할 수 없는 부분(ex. 외부 라이브러리)까지 묶어 검증할 때 사용합니다. 이는 DB에 접근하거나 전체 코드와 다양한 환경이 제대로 작동하는지 확인하는데 필요한 모든 작업을 수행할 수 있습니다. 그러나, 통합 테스트가 응용 프로그램이 완전하게 작동하는 걸 무조건 증명하지는 않습니다.

통합 테스트의 장점은 단위 테스트에서 발견하기 어려운 버그를 찾을 수 있다는 점입니다. 예를 들어, 통합 테스트에서는 환경 버그(ex. 싱글 코어 CPU에서는 잘 실행되나 쿼드 코어 CPU에서는 잘 실행되지 않음)이 발생할 수 있습니다.

한편, 통합 테스트의 단점은 단위 테스트보다 더 많은 코드를 테스트하기 때문에 신뢰성이 떨어질 수 있다는 점이다. 또, 어디서 에러가 발생했는지 확인하기 쉽지 않아 유지보수하기 힘들다는 점도 있다.

</div></details>

## Unreal Automation Test

[자동화 시스템 개요](https://docs.unrealengine.com/4.27/ko/TestingAndOptimization/Automation/)

**자동화 시스템은 Functional Testing Framework (펑셔널 테스팅 프레임워크) 기반으로 만들어졌으며, 하나 이상의 자동화 테스트를 수행하는 식으로 이루어지는 게임플레이 레벨 테스트를 위해 디자인된 것**입니다. 작성되는 대부분의 테스트는 펑셔널 테스트, 로우 레벨 코어 또는 에디터 테스트로, 자동화 프레임워크 시스템을 사용하여 작성해야 합니다.

* 테스트 유형을 참고할 수 있습니다. 단 테스트 유형과 `ex) SmokeFilter`, `ex) ApplicationContextMask`는 다르게 되어 있습니다.

<details><summary>테스트 디자인 지침</summary>
<div markdown="1">

게임 또는 프로젝트를 테스트할 때, 에픽의 자동화 테스트 기준으로 삼는 몇 가지 일반적인 지침은 다음과 같습니다:

1. 게임 또는 에디터 상태를 가정하지 않습니다. 테스트는 순서 없이 또는 여러 머신에 걸쳐 병렬 실행될 수 있습니다.
2. 디스크의 파일은 찾은 상태 그대로 놔둡니다. 테스트에서 파일을 생성한 경우, 그 테스트가 완료되면 삭제합니다. (앞으로 이러한 유형의 생성 파일을 자동 삭제하도록 하는 옵션이 추가될 수 있습니다).
3. 테스트는 지난 번 실행된 이후 나쁜 상태에 있었다 가정합니다. 테스트 시작 전 생성 파일을 삭제하는 습관을 들이는 것이 좋습니다.

</div></details>

<details><summary>테스트 만들기</summary>
<div markdown="1">

[자동화 테크니컬 가이드](https://docs.unrealengine.com/4.27/ko/TestingAndOptimization/Automation/TechnicalGuide/)

가장 간단하다고 생각되는 테스트 구현 예시로 다음이 있습니다.
```cpp
IMPLEMENT_SIMPLE_AUTOMATION_TEST(TClass, PrettyName, TFlags)
```

* Flags는 어디서 테스트를 하는 것이 적합한지를 나타냅니다. 정의부를 보면 설명이 나와있습니다.

`IMPLEMENT_SIMPLE_AUTOMATION_TEST`를 사용하는 예시는 다음과 같습니다.

```cpp
#include "Misc/AutomationTest.h"

IMPLEMENT_SIMPLE_AUTOMATION_TEST(FTestClassName, "Sample.AutomationSectionClasses", EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)

bool FTestClassName::RunTest(const FString& parameters)
{
    UWorld* world = FAutomationEditorCommonUtils::CreateNewMap();
    {
        ATheFestivalCharacter* hero = world->...
    }
}
#endif
```

* `EAutomationTestFlags::ApplicationContextMask`가 없을 경우, 
    - error C2338: AutomationTest has no application flag.  It shouldn't run.  See AutomationTest.h.
* `..._LATENT_AUTOMATION_COMMAND`은 여러 프레임에 걸쳐 실행되야 하는 경우, 사용하는 잠복명령 입니다. 자동화 테크니컬 가이드를 참고하세요.

</div></details>

<details><summary>테스트 코드 분리하기</summary>
<div markdown="1">

테스트 코드를 분리하기 위해서,
1. 플러그인을 이용하는 방법
2. 매크로를 이용하는 방법이 있습니다.

플러그인을 이용하는 방법으로는 Editor전용 플러그인을 만들어서 사용합니다.
```json
	"Modules": [
		{
			"Name": "HorrorCoreEditor",
			"Type": "Editor",
			"LoadingPhase": "Default"
		}
	]
```

매크로를 이용하는 방법으로는 다음과 같이 감싸, 에디터가 아닐 경우 번역 단위에서 제외합니다.
```cpp
#if WITH_EDITOR
...
#endif
```

</div></details>

<details><summary>테스트 케이스 CSV로 관리하기</summary>
<div markdown="1">


```cpp
#include "Engine/DataTable.h"
...

USTRUCT(BlueprintType)
struct FInventory2DInsertTest : public FTableRowBase
{
	GENERATED_BODY()
    ...
```

상속받은 스트럭트로 데이터 테이블을 만듭니다.

## 데이터 테이블을 읽고, automation에 등록하기
다음과 같이 Asset을 읽어 옵니다.
```cpp
	UDataTable* TestDataTable = LoadObject<UDataTable>(nullptr, TEXT("DataTable'/HorrorCoreEditor/Horror2DInventoryTest/Horror2DInventoryTestDataTable.Horror2DInventoryTestDataTable'"));
	TestNotNull("TestDataTable is not valid.", TestDataTable);
	if (!TestDataTable)
	{
		return false;
	}
```

## automation을 이용해서 테스트 돌리기
테스트 케이스를 입력하고, 테스트를 돌리면, 테스트 결과를 얻을 수 있습니다.

</div></details>
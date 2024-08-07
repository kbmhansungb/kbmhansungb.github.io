---
layout: post
title: Good habits and mindset
---

<details><summary>좋은 코드를 많이 봐라</summary>
<div markdown="1">

훌량한 개발자는 좋은 코드를 작성해야 합니다. 좋은 코드의 조건은 말하는 사람에 따라 다르겠지만, 로직이 명확하게 드러나고, 가독성이 좋은 코드라면 좋은 코드라고 할 수 있습니다. 학교에서는 혼자 개발하거나 소규모 개발을 위주로 합니다. 따라서 발로 코드를 작성해도 프로젝트에 주는 악영향이 제한적입니다.

하지만 많은 사람들이 보는 코드에 발로 짜놓은 코드를 **커밋**해놓으면, 발로 맞는 수가 있습니다. 가독성이 떨어지는 코드는 로직을 명확하게 이해하기 힘들고, 유지보수를 어렵게 합니다. 따라서 깔끔한, 좋은 코드를 작성하는 습관이 중요합니다. (그래서 개발이 끝나고 리팩토링 과정에 적지 않은 시간을 보내는 경우가 있습니다. 동작만 하는 코드가 아니라 유지보수가 편한 코드를 작성해야 합니다.)

일반적으로 오픈소스 프로젝트, 리눅스 소스코드 등을 보면 좋은 코드의 예를 잘 알 수 있습니다. 혹은 회사에서 일하는 코드 중에서도 좋은 코드를 짜놓는 경우가 있으면 보고 배울 수 있습니다.

</div></details>

<details><summary>겸손하라</summary>
<div markdown="1">

겸손은 개발자에게 있어서 최고의 미덕입니다. 자존심이 아니라 로직을 다루기 때문입니다. '나는 항상 옳다'라는 생각을 가지고 있다면, 설계 사항에 대한 토론에서 최고의 결론을 도출해 낼 수가 없습니다. 반면 '나도 틀릴 수 있다'를 기본으로 깔고 일을 하면 스스로도 발전할 수 있고, 좀더 탄탄한 아키텍쳐가 나오게 됩니다.

겸손하지 못 하고, 고집을 부리면 프로젝트 전체가 위험에 빠질 수 있습니다. **그로인해 잘 못 된 아키텍처를 고집하다간 되돌리기 힘든 시간적 비용적 손실을 감수해야 할지도 모릅니다.**

</div></details>

<details><summary>은퇴 할 때까지 공부하라</summary>
<div markdown="1">

IT업계는 기술의 트렌드가 가장 빨리 변하는 분야 중 하나입니다. 지금까지 알고 있던 것이 한순간에 쓸모없는 기술이 될 수 있습니다. 고급 개발자가 되려면, 계속해서 고급개발자로 남고 싶다면 계속해서 신기술에 민감하게 반응하고, 공부해야 합니다.

비슷한 관심사가 있는 사람들끼리 교류를 자주하며, 스터디도 진행하면 좋습니다. 개인적으로는 오픈소스 기술 몇 개를 정해서 사용해보고, 기여해보고, 토론해보는게 좋다고 생각합니다.

물론 이것저것 찔러보는 식의 공부는 안하느니만 못 합니다. 개발자들도 이제 분업화가 되어 있습니다. 한가지 분야에 일단은 전문가가 되는 것이 중요합니다. 전문가각 되고, 그 전문분야에서 인접한 다른 분야를 공부해 나가면 될 것입니다.

그래서 중요한게 영어입니다. 인터넷을 이용해서 얻을 수 있는 최신 기술들은 대부분 영어로 작성되어 있습니다. 영어를 할 수 있는 것과 그렇지 못 한 것의 차이는 정보력에서 나타난다고 할 수 있습니다. 적어도 문장을 독해 할 수 있는 정도의 영어 실력은 갖춰야 합니다.

</div></details>

<details><summary>커뮤니케이션 능력을 키워라</summary>
<div markdown="1">

의외로 많은 개발자들이 커뮤니케이션 능력이 부족합니다. 커뮤니케이션이라 하면, 대화기술을 비롯해서 문서 작성 기술, 이메일 기술까지 모두 포함입니다.

내가 생각하는 아키텍처와 로직을 이해시키지 못 하면 고급 개발자가 될 수 없습니다. 대화없이 돌아가는 프로그램만 만들면 되는게 아니기 때문입니다. 설득 할 수 있는 능력도 개발 능력입니다.

</div></details>

<details><summary>좋은 질문 하기</summary>
<div markdown="1">

[How do I ask a good question?](https://codereview.stackexchange.com/help/how-to-ask)

먼저, 질문의 의도대로 작동하지 않는 코드 또는 실제 코드가 아닌 경우 off-topic으로 질문이 닫히게 됩니다. 프로그래밍에 관련된 다른 질문이라면 [meta](https://codereview.stackexchange.com/help/how-to-ask)에 질문하는 것이 좋습니다.

* [질문의 의도대로 작동하지 않는 코드](https://codereview.meta.stackexchange.com/questions/3649/my-question-was-closed-as-being-off-topic-what-are-my-options/3650#3650)란 다른 종속성이 포함되어 있어서 컴파일 할 수 없거나, 실행 실패(충돌, 예외 발생, 코어 덤프, 세그폴트 등)의 상황, 분명히 잘못된 경과를 생성하는 경우(단위 테스트 케이스 실패 등)을 말합니다. 코드가 올바르게 작동할 때 훨씬 더 생산적일 수 있습니다.
    - 올바르게 작동하지만, 느려서 제대로 작동하지 않는 경우 "time-limit-exceed"태그를 이용할 수 있습니다.

* Stack Exchange사이트와는 달리 Code Review는 코드에 특정한 맞춤형 조언을 제공을 목표로 합니다. 그렇게 하기 위해서는 실제 코드가 필요합니다. [질문이 실제 코드가 아닌 경우](https://codereview.meta.stackexchange.com/questions/3649/my-question-was-closed-as-being-off-topic-what-are-my-options/3652#3652)는 의사 코드나 예제 코드(의미 있는 검토를 하기에는 너무 가설적인)로 간주되지 않아야 합니다.
    - 사용하는 언어를 선택하고 실제 작동하는 코드를 올려야 합니다.

* 당연한 이야기지만 도덕적, 실용적 및 법적 이유로 직접 작성한 코드만 컴토될 수 있습니다.

* 소프트웨어 엔지니어링에 관련된 질문인 경우 [softwareengineering](https://softwareengineering.stackexchange.com/)에 올리도록 합시다.

</div></details>

<details><summary>코드 질문 템플릿</summary>
<div markdown="1">

모든 질문에 적합하지는 않지만, 코드리뷰를 올릴 떄 참고하기 위해서 정리한 내용입니다.

* 제목: 코드의 목표를 요약합니다. ex) 스윙 UI가 있는 인생 게임, 금연 iOS 앱용 뷰 컨트롤러
* 세부 정보: 누구나 알아 들을 수 있도록 명확하게 명시되어야 합니다. 즉 세부 정보가 구체적으로 제시되어야 합니다.
* 코드블럭 네임: 알아보기 쉽도록 합시다.   
    ```
    코드: 코드가 어떤 동작을 하는지 구체적인 설명이 있으면, 보다 좋은 리뷰를 받을 수 있습니다.
    ```
* 테스트 및 동작: 이부분은 아직 잘 모르므로 생략합시다.
* 태그: 코드의 배경을 제시할 필요가 있습니다. 챌린지의 경우는 (programming-challenge), 새로운 프로그래밍 언어를 배우고 있을때는 (beginner), 공용 라이브러리 기능을 연습용으로 의도적으로 다시 구현하는 경우(reinventing-the-wheel)과 같이 제공해야 합니다.

</div></details>

### 좋은 습관들

<details><summary>프로그래밍 책 읽기</summary>
<div markdown="1">

프로그래밍 책을 읽는 것은 강장 기본적인 습관입니다. 새로운 언어를 배우거나 부족한 개발 실력을 채울 때 가장 많이 사용되는 방법입니다. 프로그래밍 책을 읽는 것을 습관화한다면 여러분은 새로운 언어를 배울 때 빠르게 습득이 가능합니다.

프로그래밍 책에는 프로그래밍 언어, 프로젝트 아키텍처, 모범 사례, 다양한 기술 및 이론에 대한 책을 포합합니다. 또한, 이펙티브 엔지니어와 같은 개발자를 위한 책을 읽게 되면 여러분은 개발 실력을 올리기 위한 기초 공사가 끝납니다.

</div></details>

<details><sumary>튜토리얼 보기</summary>
<div markdown="1">

처음 언어를 배울때는 튜토리얼을 따라서 만듭니다. 그래야 개발 언어에 대한 접근성이 올라가고, 새로운 프로젝트를 하는데 자신감을 가질 수 있습니다. 튜토리얼을 보는 습관을 가지면 어떤 언어를 배우던, 실력 있는 프로그래머가 될 수 있습니다. **프로그램을 구축하기 위해 사용한 기술이 동작하는 방법과 새로운 작업을 수행하는 방법을 이해하는데 도움이 됩니다.**

</div></details>

<details><summary>매일 기능 만들기</summary>
<div markdown="1">

매일 코딩을 통한 기능을 만듭니다. 메뉴얼과 강의, 책을 통해서 먼저 코딩을 시작합니다. 먼저 부담을 없애가 코딩하는 것에 거부감이 줄어듭니다. 또한 하다보면 욕심이 생겨서 개인 프롲게트를 진행하며 새로운 기능을 기존 프로젝트에 적용해볼 수 있습니다. **하나씩 기능을 더하고, 코드를 정리하다 보면 실력이 엄청나게 증가하게 됩니다. 매일 기능 만들기 습관은 가장 추천하는 습관입니다.**

</div></details>

<details><summary>면접 준비하기</summary>
<div markdown="1">

프로그래밍 직업은 개발자로서 배우고 성장하는 직무입니다. 직무를 키우기 위해서는 면접 준비를 하면서 기본적인 프로그래밍 개념과 문제 해결 능력을 키워야 합니다.

학교에서 이론과 문제 해결 능력을 키웠지만 학교를 졸업하면 프로그래밍 개념과 문제 해결 능력을 시험받을 기회가 없습니다. 이를 면접을 준비하면서 키울 수 있습니다. 면접 준비를 습관처럼 한다면 언제든 이직할 수 있다는 마음에 안정도 가질 수 있습니다.

</div></details>

<details><summary>수도 코드 작성하기</summary>
<div markdown="1">

먼저 코드를 작성하기보다 pseudo code를 작성하여 어떻게 코드를 작성할지 설계를 해봅니다. **이 습관은 좋은 코드를 작성하는데 매우 좋은 습관입니다. 작업을 시작하기 전 코드를 기획하는 습관을 키워야 합니다.**

</div></details>

<details><summary>더 나은 네이밍을 사용하자</summary>
<div markdown="1">

네이밍 습관은 유지보수와 버그 처리에 매우 도움이 되는 습관입니다. 네이밍을 이상하게 하는 살마과 일을 하거나, 전 사수가 그런 코드를 작성했다면 유지보수가 매우 어려울 것 입니다. 본인이 만든 코드도 이해가 안 되는 경우도 있습니다. 그렇기 때문에 좋은 네이밍 습관은 시니어 개발자로 올라가기 전 꼭 가져야 할 습관입니다.

</div></details>
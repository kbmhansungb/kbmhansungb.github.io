---
layout: post
title: INLINE
---


## 언제 FORCEINLINE을 사용하는가?
프로파일링을 하고 도움이 될 것으로 생각되는 핫 코드가 있는 경우가 아니면 사용하지 마십시오. 그런 다음 프로파일링하고 실제로 도움이 되는지 확인하십시오.

그리고 네, 필요하다고 느낄 때까지 일반 기능을 사용하기로 결정했습니다. 필요하지 않은 경우 자습서에 포함시키는 것이 너무 이상해 보이기 때문에 이유가 있는지 궁금합니다.

.h 파일의 getter 및 setter에만 사용합니다.

inline이 뭔지도 모르고 쓰고 있었네,,,

https://thinkpro.tistory.com/140


## FORCEINLINE_DEBUGABLE
https://usagi.hatenablog.jp/entry/2017/06/14/152825

FORCEINLINE_에 대해서 이해도가 생기면 사용하자. 당장은,,, 모르겠다.

* 마이크로소프트로 가면 디스어셈블리 뭐시기 하는데,,, 어지럽고..
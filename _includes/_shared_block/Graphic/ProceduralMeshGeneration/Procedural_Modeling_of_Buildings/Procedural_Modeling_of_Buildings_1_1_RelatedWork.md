절차적 모델링은 Semi-Thue 공정[Davis et al. 1994], 촘스키 문법[Sipser 1996], 그래프 문법[Ehrig et al. 1999], 형태 문법[Stiny 1975] 및 귀속 문법[Knuth 1968]. 그러나 생산 시스템의 단순한 사양은 기본에 불과합니다. 기하학적 해석, 간결한 표기법, 유도 제어, 실제 모델 설계와 같은 몇 가지 질문은 여전히 ​​해결해야 합니다.
<!-- Procedural modeling can draw from a large spectrum of production systems such as Semi-Thue processes [Davis et al. 1994], Chomsky grammars [Sipser 1996], graph grammars [Ehrig et al. 1999], shape grammars [Stiny 1975], and attributed grammars [Knuth 1968]. However, the mere specification of a production system is only the basis. Several questions, such as the geometric interpretation, concise notation, control of the derivation, and the design of actual models, still have to be addressed. -->

식물의 기하학적 모델링을 위해 Prusinkiewicz와 Lindenmayer는 L-시스템을 사용하여 로고 스타일 거북이로 해석되는 문자열을 생성함으로써 놀라운 결과를 얻을 수 있음을 보여주었습니다[Prusinkiewicz and Lindenmayer 1991]. L-시스템은 거북이 위치를 쿼리하도록 확장되었습니다[Prusinkiewicz et al. 1994], 일반 컴퓨터 시뮬레이션[Mech and Prusinkiewicz 1996], 자기 민감성[Parish and Muller 2001] 및 사용자 생성 곡선[Prusinkiewicz et al. 2001].
<!-- For the geometric modeling of plants, Prusinkiewicz and Lindenmayer showed that wonderful results can be achieved by using L-systems to generate strings that are interpreted with a LOGOstyle turtle [Prusinkiewicz and Lindenmayer 1991]. L-systems have been extended to query the turtle position [Prusinkiewicz et al. 1994], to incorporate general computer simulation [Mech and Prusinkiewicz 1996], self-sensitivity [Parish and Muller 2001], and ¨ user generated curves [Prusinkiewicz et al. 2001]. -->

아키텍처에서는 문법을 형성합니다[Stiny 1975; Stiny 1980]은 건축 설계의 구성 및 분석에 성공적으로 사용되었습니다[Downing and Flemming 1981; Duarte 2002; 플레밍 1987; 코닝과 아이젠버그 1981; 스티니와 미첼 1978]. 모양 문법의 원래 공식은 레이블이 지정된 선과 점의 배열에서 직접 작동합니다. 그러나 파생은 본질적으로 복잡하며 일반적으로 적용할 규칙을 결정하는 사람과 함께 수동 또는 컴퓨터로 수행됩니다. 모양 문법은 문법을 설정하기 위해 단순화될 수 있습니다[Stiny 1982; Wonkaet al. 2003] 컴퓨터 구현에 보다 쉽게 ​​적응할 수 있도록 합니다. 셀룰러 텍스처 [Legakis et al. 2001] 벽돌 패턴을 계산하는 데 사용할 수 있으며 생성 메시 모델링은 단순한 표면에서 복잡한 매니폴드 표면을 생성할 수 있습니다[Havemann 2005].
<!-- In architecture, shape grammars [Stiny 1975; Stiny 1980] were successfully used for the construction and analysis of architectural design [Downing and Flemming 1981; Duarte 2002; Flemming 1987; Koning and Eizenberg 1981; Stiny and Mitchell 1978]. The original formulation of the shape grammar operates directly on an arrangement of labeled lines and points. However, the derivation is intrinsically complex and usually done manually, or by computer, with a human deciding on the rules to apply. Shape grammars can be simplified to set grammars [Stiny 1982; Wonka et al. 2003] to make them more amenable to computer implementation. Cellular textures [Legakis et al. 2001] can be used to compute brick patterns and generative mesh modeling can generate complex manifold surfaces from simpler ones [Havemann 2005]. -->

문법으로 정의된 프레임워크는 절차적 모델링의 필수 부분이지만 아키텍처 구성을 생성하는 규칙을 추상화하는 것이 필요합니다. 이 작업을 위해서는 더 큰 라이브러리가 필요하지만 시각적 사전 [Ching 1996], Mitchell의 "The Logic of Architecture" [1990], "Space Syntax" [Hillier 1996], 디자인 패턴 [Alexander et al. 1977], 대칭 연구[March and Steadman 1974; Shubnikov 및 Koptsik 1974; 웨일 1952].
<!-- While the framework defined by the grammar is one essential part of procedural modeling, it is then necessary to abstract rules that create architectural configurations. While a larger library is necessary for this task, we would recommend starting with books that emphasize structure of architecture, such as a visual dictionary [Ching 1996], “The Logic of Architecture” by Mitchell [1990], “Space Syntax” [Hillier 1996], Design patterns [Alexander et al. 1977], and studies of symmetry [March and Steadman 1974; Shubnikov and Koptsik 1974; Weyl 1952]. -->
---
layout: post
title: Math
---

## 벡터 공간
<!-->

평행하지 않은 두 공간벡터 $ \overrightarrow{u_1},\overrightarrow{u_2} $와  서로 다른 두 점 $ P_1, P_2 $에 대하여 두 직선이 다음과 같이 주어져 있을 때

<center>

$
g_1 : \overrightarrow{OP_1} + \overrightarrow{tu_1},;
 g2 : \overrightarrow{OP_2} + \overrightarrow{tu_2}
$

</center>

이 때 두 직선사이의 거리는 

<center>

$
\tfrac
{|\overrightarrow{P_1P_2} \cdot (\overrightarrow{u_1} \times \overrightarrow{u_2})|}
{|\overrightarrow{u_1} \times \overrightarrow{u_2}|}
$

</center>

<-->


- [ ] [언리얼 엔진 피직스 에센셜](https://www.perlego.com/book/4136/unreal-engine-physics-essentials-pdf)

## 에라토스테네스의 체 (소수 구하기)
프로그래밍 문제를 해결하다 보면 소수를 활용해야 하는 경우가 종종 발생하는데, 그 중에서도 특정한 값 N이하의 소수를 모두 찾아야 하는 경우가 꽤 많습니다. 그럴 때 초, 중학교 시절 노가다 취급했던 에라토스테네스의 체가 꽤 좋은 방법이 되어줍니다.

[에라토스테네스의 체 설명](https://www.weeklyps.com/entry/%EC%97%90%EB%9D%BC%ED%86%A0%EC%8A%A4%ED%85%8C%EB%84%A4%EC%8A%A4%EC%9D%98-%EC%B2%B4-%EC%86%8C%EC%88%98-%EA%B5%AC%ED%95%98%EA%B8%B0)

## 라디안(radian)과 디그리(degree)

$$
f(x) = \int_{-\infty}^\infty \hat f(\xi)\,e^{2 \pi \xi x} \,d\xi
$$

https://darkpgmr.tistory.com/26

우리가 일상적으로 사용하는 각의 단위는 디그리(degree)입니다. 즉, 원 한바퀴를 360도로 표현하는 방법입니다. 반원은 180도, 직각은 90도 등 degree는 우리에게 매우 익숙한 각의 단위입니다.

그런데, 각을 표현하는 다른 방법으로 라디안(radian)이 있습니다. 보통 라디안은 부채꼴을 중심각을 가지고 설명되는데, 아래 그림과 같이 호의 길이가 반지름과 같게 되는 만큼의 각을 1라디안(radian)이라고 정의합니다.

* l = rad * r
    - 정의에 따르면 호의 길이는 rad * 반지름임을 알 수 있습니다.

정의에 따르면 왠지 라디안은 반지름에 대한 상대적인 각도의 단위처럼 생각됩니다. **하지만 radian은 degree처럼 절대적인 각도의 단위입니다.** 실제로 1 radian은 약 57.3도에 해당하는 각입니다. 그러면 2 radian은 약 114.6도가 됩니다. 여기서 우리는 degree 보다는 radian이 훨씬 큰 각의 단위라는 걸 알 수 있습니다.

* 친숙한 degree로만 각을 표현하면 좋을텐데 왜 이렇게 복잡하게 radian이라는 것을 도입해서 문제를 어렵게 하는 걸까요?

degree와 radian의 변환 관계는 다음과 같습니다. 앞으로 가면 제자리로 돌아오는 링을 생각하면 다음과 같습니다. 

* 1 degree : 360 degree = 1 rad : 2 PIE rad  
    - 1 radian = 180 / PIE degree 

## 치트 북

수식을 찾느라 고통받을 때, 다음의 방법으로 도움받을 수 있다고 생각합니다.

수학 공식이 필요하다면 [Theoretical Computer Science Cheat Sheet](https://tug.org/texshowcase/cheat.pdf)에서 볼 수 있습니다.

# Gradient(그라디언트)

어떤 다변수 함수 f(x1, x2, ... xn)이 있을 때, f의 gradient는 ∇f = (∂f/∂x1, ∂f/∂x2 ... ∂f/∂xn)입니다.

즉, gradient는 위 식과 같이 각 변수로의 일차 편미분 값으로 구성되는 벡터입니다. 그리고 이 벡터는 **f의 값이 가장 가파르게 증가하는 방향을 나타냅니다.**

이러한 gradient의 특성은 어떤 함수를 지역적으로 선형근사(linear approximation)하거나 혹은 gradient descent 방법(Gradient Descent 탐색 방법 글 참조)처럼 함수의 극점(최대값, 최소값 지점)을 찾는 용도로 활용될 수 있습니다.

# Jacobian(야코비언) 행렬

Jacobian은 어떤 다변수 벡터함수에 대한 일차 미분으로 볼 수 있습니다.

앞서 나온 gradient나 jacobian이나 모두 함수에 대한 일차 미분을 나타낸다는 점에서 돌일합니다. 다만 그레디언트는 다변수 스칼라 함수(scalar-valued function of multiple variables)에 대한 일차 미분인 반면 jacobian(야코비언)은 다변수 벡터 함수(vector-valued function of multiple variables)에 대한 일차미분입니다.

즉, 그레디언트는 통상적인 일변수 함수의 일차미분을 다변수 함수로 확장한 것이고, jacobian(야코비언)은 이를 다시 다변수 벡터함수로 확장한 것 입니다.

* **벡터 함수에 대한 일차 미분입니다.**

Jacobian(야코비언)이나 그레디언트나 모두 함수에 대한 일차미분이기 때문에 미분이 가지고 있는 의미나 성질은 모두 동일하게 적용됩니다. 즉, 어떤 함수의 지역적인 변화 특성을 파악할 때, 지역적인 함수의 변화를 선형근사 할 때 또는 함수의 극대(극소)를 찾을 때 활용될 수 있습니다.

* 선형근사 한다는 예시를 좀더 이해필요

* 이거는 더 많은 예시를 찾아봐야 할 것 같습니다.

자코비안은 매핑함수로 각 조인트 각도를 ee

읽을 순서들

http://t-robotics.blogspot.com/2013/12/jacobian.html#.YtEw_3ZBxD8

https://darkpgmr.tistory.com/132?category=460967

https://blog.naver.com/PostView.naver?blogId=nswve&logNo=222290664934&parentCategoryNo=&categoryNo=48&viewDate=&isShowPopularPosts=false&from=postView

https://blog.daum.net/pg365/97

# Hessian(헤시안) 행렬

gradient, jacobian이 모두 함수에 대한 일차미분을 나타내는 반면 Hessian은 함수의 이차미분을 나타낸다는 점에서 차이가 있습니다.

즉, hessian은 함수의 곡률(curvature) 특성을 나타내는 행렬로서 최적화 문제에 적용할 경우 Hessian을 이용하면 다음 식과 같이 p긑처에서 함수를 2차 항까지 근사시킬 수 있습니다.(second-order Taylor expansion)

# Laplacian(라플라시안) 행렬

https://m.blog.naver.com/sallygarden_ee/221287818061

# TRS행렬과 트랜스폼(Transform)
https://velog.io/@ounols/%EA%B2%8C%EC%9E%84-%EC%88%98%ED%95%99-1.-3%EC%B0%A8%EC%9B%90-%EA%B3%B5%EA%B0%84-%EB%A7%9B%EB%B3%B4%EA%B8%B0

https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=destiny9720&logNo=221409628172 

https://www.dogdrip.net/152647379 

https://www.dogdrip.net/index.php?document_srl=149009704&mid=dogdrip&cpage=2 

# 강체
**강체란 물리학에서 형태가 변하지 않는 물체를 가리킨다.** 외력이 가해져도 크기나 모양이 변형되지 않거나 변형되는 정도가 무시할 수 있을 만큼 작다면 강체로 가정하기도 한다.

## 회전운동과 병진운동
회전운동이란 물체가 한 점을 축으로 회전하는 운동을 말하며 병진운동이란 평행이동 즉 질점계의 모든 질점이 똑같이 이동하는 운동을 말한다. 강체의 가장 일반적인 운동은 질량중심의 병진운동과 질량중심을 지나는 축에 대한 회전운동의 결합이다. 이 운동을 설명 할 때 기본적으로 두 가지의 조건이 만족 되어야 하는데, 첫 번째는 회전축은 물체의 대칭축 이어야 하고, 물체의 질량 중심이 회전축 상에 있어야 한다. 두 번째는 회전축이 움직이더라도 그 방향은 달라져서는 안 된다.

앞서 말한바와 같이 강체의 운동은 병진운동과 회전운동의 합으로 나타낼 수 있다. 예를 들면, 달리는 차의 바퀴를 생각해보자. 질량 중심의 속도는 v, v' 벡터들은 각 지점에서의 질량중심에 대한 상대속도를 나타내며 이것은 중심축에 대한 회전운동의 속도를 말한다. 움직이는 바퀴에 대해 복합적인 운동을 관성계에서 보면, 각 호부분의 실제 속도는 두 벡터의 합이 된다. 바퀴가 지면과 접하고 있는 점은 순간적으로 그 속도가 0이 된다. 또 바퀴의 맨 윗부분은 속도가 질량중심 속도의 2배가 되고, 양 옆의 점들의 속도는 수평면과 45도 각도를 이루고 있음을 알 수 있다.

<center>

![Image](https://astro.kasi.re.kr/file/astro_page/496235600088054.jpg)

</center>

## 돌림힘(Torque, 토크)
토크, 즉 돌림힘이라고 하는 녀석은 강체의 원운동에서의 힘이라고 하는 녀석과 함께 생각해볼 수 있습니다. 

문을 닫을 때 문이 잘 닫히는 방향은 축에서부터 멀리 떨어질 때이며, 물체는 축을 중심으로 회전하므로, 축에서 작용점까지의 길이는 반지름 r으로 표현할 수 있습니다. 여기서 작용하는 힘을 돌림힘 이라고 해석할 수 있습니다. 기호로는 그리스 문자인 타우(τ)를 사용할 수 있습니다.

토크가 크다는 말은, 힘이 잘 작용하여 문이 잘 돌아간다는 뜻이고 토크가 작다는 말은 힘이 잘 작용하지 않아서, 문이 잘 돌아가지 않는다는 의미입니다. 

토크라고 하는 요소는 반지름의 길이에 비례하면 할 수록, 힘에 비례할 수록 쉽게 닫을 수 있으닌까 다음과 같이 적을수 있습니다. τ∝r, τ∝F

똑같은 힘이 작용하더라도 그 힘이 어느 방향으로 작용하느냐에 따라 달라집니다. τ∝sinθ

토크는 다음과 같이 정의됩니다. τ=rFsinθ

* **강체의 회전 운동은 선형 운동의 어떤 것과 대응이 됩니다.**https://blog.naver.com/PostView.naver?blogId=at3650&logNo=220109580023&parentCategoryNo=&categoryNo=10&viewDate=&isShowPopularPosts=false&from=postView

돌림힘에서 힘요소는 (관성모멘트) X (각가속도)를 나타내는 Ia = mr^(2)a로 표시할 수 있습니다.


https://blog.naver.com/PostView.naver?blogId=at3650&logNo=220102763738&parentCategoryNo=&categoryNo=10&viewDate=&isShowPopularPosts=false&from=postView

**토크는 벡터량으로서**, T = r X F로 표현할 수 있습니다. 이를 다시 말하면 ||r X F|| = ||r|| ||F|| sinθ 로 표현할 수 있습니다.

https://gafferongames.com/post/physics_in_3d/

## 원하는 회전 방향으로 물체를 회전시키는 토크를 찾는 방법



[Link](https://godotengine.org/qa/92316/how-to-find-torque-to-rotate-object-towards-desired-rotation)

## 정적 평형과 동적 평형

## 각 운동량(Anular Momentum)

https://blog.naver.com/PostView.naver?blogId=at3650&logNo=220234434194&redirect=Dlog&widgetTypeCall=true&topReferer=https%3A%2F%2Fwww.google.com%2F&directAccess=false 

# 게임 엔진
게임 물리 엔진을 기본적으로 **고전역학**을 다룬다. 고전역학은 물체에 작용하는 힘과 운동의 관계를 설명한다. 즉, 물리엔진은 게임 속 물체에 힘이 작용했을 때 물체의 운동을 시뮬레이션하는 역할을 한다. 물체에는 형태가 고정되어 변하지 않은 강체(Rigid Body)와 형태가 변형되는 연체(Soft Body)가 있는데, 물리엔진에서는 대부분 강체만 다룬다.

강체의 운동은 크게 병진 운동(Translational Motion)과 회전 운동(Rotational Motion)으로 나뉜다. 병진 운동은 강체의 모든 부분이 한 방향으로 움직이는 것이다. 병진 운동의 결과로 물체의 위치가 변하게 된다. 물체가 얼마나 빠르게, 또 어떤 방향으로 이동하는지 나타내는 물리량이 속도, 시간에 따라 속도가 변하는 정도를 나타내는 물리량이 가속도이다. 

## 회전 운동
회전 운동은 어떠한 기준 축을 중심으로 물체가 도는 운동이다. 

## 병진 운동
병진운동으로 물체의 위치가 변하듯 회전운동을 통해서 물체의 각도가 변한다. 속도와 마찬가지로 물체의 각도가 얼마나 빠르게 변했는지를 나타내는 물리량이 각속도, 각속도가 얼마나 빠르게 변하는지를 나타내는 물리량은 각가속도이다.

이제 힘과 회전운동의 관계를 살펴보면 **회전운동에서 물체를 회전시키게 하는 힘은 토크이며 뉴턴의 운동 2법칙과 형태와 비슷하게 이는 관성 모멘트와 각가속도의 곱으로 나타낼 수 있다.** 관성 모멘트는 물체를 이루는 입자들의 질량과 회전축과의 거리 제곱의 합이다. 관성 모멘트는 질량과 같이 단일 수치로 나타낼 수 없고 회전축에 따라서 달라진다. 회전축을 특정하면 단 하나의 값이 나오지만 회전축은 임의로 잡는 것이 가능하다. 모든 관성 모멘트의 값을 관성 텐서(Inertia tensor)라는 행렬에 넣는 것이 가능하다. 3차원 공간에서 강체의 관성 텐서는 3*3 행렬이며 대각선 방향 원소는 X축, Y축, Z축을 중심으로 회전할 때의 관성 모멘트이며, 나머지 부분은 관성곱이다. 

## 강체의 운동을 계산하는 함수 Integrate

```cpp
void RigidBody::Integrate(float duration)
{
    m_lastFrameAcceleration = m_acceleration;
    m_lastFrameAcceleration += m_forceAccum * m_inverseMAss;

    CXMFLOAT3 angularAcceleration = XMVector 3TransformNormal(m_torqueAccum, m_inverseInertiaTenserWorld);

    m_velocity += m_lastFrameAcceleration * duration;
    m_rotation += angular Acceleration * duration;

}
```

## 물리엔진의 실사용
강체는 간단한 물리 법칙을 그대로 사용할 수 있으나, 물과 같은 유체는 미적분방정식을 그대로 사용하기 어렵다. 컴퓨터는 미적분방정식을 풀 때 사람처럼 간단한 공식을 외워서 하는 것이 아니라, 미적분을 유도하는 공식부터 연산 과정에 포함시키므로 시간이 훨씬 많이 걸린다. 그래서 사용할 때에는 복잡한 수식을 간소화시켜 사용하는 경우가 많다. 수식뿐만 아니라 숫자를 간소화하기도 하는데, 가령 중력가속도를 편의상 1로 설정할 수도 있는 것이다. 실제 세계와 완벽하게 같지는 않지만 비슷하게 만들기 위함이다. 이를테면, ‘마비노기영웅전’에서는 공기저항을 아예 없앤 대신, 떨어지는 물체의 최대속도를 지정하여 속도를 제한한다. 그래서 실제로 물체가 떨어지는 것처럼 보인다.


물리엔진은 게임에서만 필요한 것은 아니다. ‘트랜스포머’ 등과 같은 영화 특수효과에서도 물리 법칙 구현은 필수적이다. 그런데 영화 특수효과에서 사용되는 물리엔진과 게임 물리엔진 사이에는 차이점이 있다. 1초에 24장의 화면을 보여주는 영화에서 고품질 영상을 얻기 위해 한 화면을 만드는 데 많은 시간이 걸려도 이를 감수한다. 반면 게임은 그 자리에서 1초에 들어갈 30장 이상의 화면을 생성해야 한다. 따라서 복잡한 물리 공식을 30분의 1초마다 풀어내는 것은 어려운 일이다. 초기 물리엔진은 포탄 탄도 계산 등 간단한 기능만을 제공했으나 지속적으로 기능을 확장했다. 덕분에 지금은 약간 어색한 수준이나마 옷과 같은 변형체, 연기와 같은 유체도 처리할 수 있다. 이는 컴퓨터 하드웨어와 소프트웨어의 발달 덕분에 가능해졌다. 특히 눈부시게 진화하는 CPU와 그래픽 카드에 힘입은 바가 크다. 복잡한 변형체와 유체 등을 처리하는 이론은 이미 ‘캐러비안의 해적’ 등과 같은 영화의 특수효과 제작에 이용되고 있다. 또 최근 공개된 디즈니 실사 영화 ‘라이언킹’에서는 유니티 엔진으로 세트 및 기본 애니메이션을 생성하고, 배우의 목소리 연기를 디지털 캐릭터와 동기화했다.


물리엔진은 PC의 탄생과도 밀접한 관계가 있다. 최초의 컴퓨터 에니악(ENIAC)은 탄도 계산을 위해 개발 됐다. 수학자들이 지도와 각도기로 했던 탄도 계산을 에니악이 처리했다. 당시 에니악이 계산했던 탄도의 궤적은 이제 스마트폰으로도 쉽게 측정할 수 있는 수준이지만, 시간이 흘렀기 때문에 더 복잡한 계산을 해야하는 SW가 필요하게 되고, 이 SW를 물리엔진이 담당하고 있다. 1980년대부터 슈퍼컴퓨터가 등장해 대기와 해류, 날씨 등은 시뮬레이션되어 사용되었다. 오늘날 물리엔진은 더 영역이 넓어지고 있다. 제품을 만들 때 발생할 수 있는 일들을 SW로 시뮬레이션함으로써 오차와 비용을 줄이고, 개발 기간을 단축할 수 있다. 자동차 업체와 항공기 제작업체 경우 막대한 개발 비용을 줄이기 위해 이같은 방법을 쓰고 있다.


이처럼 궁극적으로 게임 엔진은 단순한 게임 제작 도구에 머물지 않고 하나의 콘텐츠 제작 ‘플랫폼’으로 진화하는 추세다. 많은 게임 개발자들이 처음엔 ‘게임이 좋아서, 직접 만들고 싶어서’ 게임 엔진을 접하고 개발자의 길을 걷게 됐다고 말한다. 그러나 이젠 게임 엔진을 배워서 게임만 만들 수 있는 게 아니다. 활용하기에 따라 훨씬 다양한 영역에서 유용하게 사용할 수 있는 환경이 마련되고 있다. 개발에 관심 있는 사람이라면 취미로 게임 엔진 하나 정도는 입문해 보는 것도 좋을 듯하다. 

[천문학습관](https://astro.kasi.re.kr/learning/pageView/5231)
[게임의 심장, 게임엔진](https://www.ksakosmos.com/post/%EA%B2%8C%EC%9E%84%EC%9D%98-%EC%8B%AC%EC%9E%A5-%EA%B2%8C%EC%9E%84%EC%97%94%EC%A7%84)
[정보통신기술용어해설](http://www.ktword.co.kr/word/abbr_view.php?nav=2&m_temp1=4727&id=1122)
[Chapter 12](http://teacher.pas.rochester.edu/phy121/lecturenotes/Chapter12/Chapter12.html)

더 자세한 내용은 GamePhysicsCookbook에 있다고 한다.

## 강체의 회전 운동


유클리드 기하학, 비유클리드 기하학, 카테시안 좌표계, 곡선 좌표계??

# 유클리드 공간(Eculidean space)란?
이 일반화는 유클리드가 생각했던 거리와 길이와 각도, 좌표계를 도입하여, 임의차원의 공간으로 확장한 것을 말합니다.

1. 어떤 한 점에서 어떤 다른 한 점으로 선분을 그릴 수 있다.  
2. 임의의 선분을 선을 따라 다른 선분으로 연장할 수 있다.    
3. 어떤 한 점을 중심으로 하고 이에 대한 거리(반지름)로 하나의 원을 그릴 수 있다.     
4. 모든 직각은 서로 같다.   
5. 평행선 공준, 두 직선이 한 직선과 만날 때, 같은 쪽에 있는 내각의 합이 2직각(180도)보다 작으면 이 두 직선을 연장할 때 2직각보다 작은 내각을 이루는 쪽에서 반드시 만난다.   

[유클리드 공간](https://supermemi.tistory.com/75)

# 데카르트 좌표계(Cartesian coordinate system)과 오일러 좌표계(Euler coordinate system)

데카르트 좌표계(Cartesian coordinate system)는 직교좌표계라고도 하며, 임의의 차원의 유클리드 공간을 나타내는 좌표계중에 하나입니다. 직교좌표계는 다른 좌표계와는 달리 임의의 차원으로 쉽게 일반화 할 수 있습니다. 2차원의 경우 { x, y } 3차원의 경우 { x, y, z } 물리학에서 다루는 4차원의 경우 { x, y, z, t }로 표현할 수 있습니다.

# 오일러

## 오일러 각(The Euler Angle)
오일러 각도는 회전 변환 행렬이나 좌표축의 자세로 표현하는데 직관적인 방법입니다. 강체가 놓인 방향을 3차원 공간에 표기하기 위해 도입한 세 가지 각도입니다.

3차원 공간의 좌표 계를 (x, y, z)라고 하고 이를 회전시킨 좌표계를 (X, Y, Z)라고 하면, 강체의 방향은 다음의 세 각도로 표시될 수 있습니다.

* α ( or ψ, psi, 프시)
    - Z-축을 회전축으로 하여 회전된 x-y 좌표축의 각도
* β ( or θ, theta, 세타)
    - 회전된 x-축 (N축, 북극)을 회전축으로 하여 회전된 z-y 좌표축의 각도
* γ ( or φ, phi, 피) 회전된 z-축(Z축)을 회전축으로 하여 최전된 x-y 좌표축의 각도

위와 같이 하여 강체의 방향은 세 개의 각도로 표시될 수 있습니다.

* 로봇제어 에서는 (ψ, θ, φ)의 표현이 자주 사용됩니다.
* 오일러 각도는 강체의 자세를 좌표축의 회전으로 표현하는 방법 중 하나로, 회전 축의 순서에 따라 Z-X-Z 좌표라고도 불립니다.

[오일러 각도 변환](https://blog.daum.net/aero2k/56)

* 강체란 한 물체의 모든 물질 원소들이 그 물체 안의 다른 물질 원소들에 대해 이동되거나 회전될 수 없을 때 그 물체를 강체(rigid body)라고 부릅니다.
* 오일러각은 직각 좌표계(Cartesian coordinate system)에서 X, Y, Z축을 따라 오른손 좌표계 방향으로 각을 정의하고 정해진 순서에 따라 3번 회전운동을 수행하여 회전 운동을 표현합니다. 따라서 미리 회전 순서를 정해주지 않으면 오일러각은 매우 다양하게 정의 될 수 있습니다.
* 좌표계 : OpenGL과 같은 류는 오른손 좌표계를 사용합니다. 만약 Direct 3D와 같은 왼손 좌표계를 사용하고 있다면, 행렬들을 전치(transpose)해야 합니다.
* 행렬은 OpenGL처럼 열 우선방식으로 처리합니다.

## 회전각 및 축의 양의(Positive) 방향

무게중심으로부터 앞쪽으로 x축, 오른쪽으로 y축, 아래쪽으로 z축이 정의되어 있습니다. 여기서, 화살표의 방향이 각 축의 (+)양의 방향과 회전의 (+)양의 방향을 나타내게 됩니다.

* 회전의 양의 방향을 찾을 때는 오른손 법칙을 적용하면 쉽습니다.

**물체의 현재의 기울임 자세와 관계없이 원하는 어떠한 자세도 이 세번의 회번을 통해서 도달할 수 있다는 것이 핵심입니다.**

[회전각 및 축의 양의(Positive)방향](https://m.blog.naver.com/droneaje/221999534231)

[오일러 각/회전을 통한 좌표변환 공식의 유도 및 정리 - 강체의 움직임을 기준으로](https://m.blog.naver.com/droneaje/221999534231)

**1번째 회전 x축을 기준으로 한 Roll 회전**
1번째 회전은 x축을 기준으로 한 Roll회전입니다. 이를 1번째 오일러 각 (1 Euler Angle)이라고 부릅니다.

드론이 우측으로 기울 떄 +φ의 각을 갖게 되고 (x, y, z)좌표 축은 (x1, y1, z1)이라는 새로운 좌표 축을 갖게 됩니다. 물론 X축을 기준으로 회전했기 때문에, x1 축은 x축과 동일합니다.

이의 변환행렬은 다음과 같습니다.

```
| x_1 |   | 1    0     0  || x |
| y_1 | = | 0  cosφ  sinφ || y |
| z_1 |   | 0  sinφ  cosφ || z |
```
* x_1 = x * 1 + y * 0 + z * 0 = x

이를 회전 행렬(Rotation Matrix)를 통해 두 좌표축 간의 관계를 나타낼 수 있습니다.

```
B_1 = R_1 B
```

R1은 Roll회전(Rotation) 행렬을 나타내고, B는 회전이 있기 전의 물체고정좌표계, B1은 x축을 기준으로 회전한 이후의 새로운 물체고정좌표계를 의미합니다.

**2번째 회전 y1축을 기준으로 한 Pitch 회전**
2번째 회전은 y1 축을 기준으로 한 Pitch 회전이고, 이를 2번째 오일러 각(2 Euler Angle)이라고 부릅니다.

드론의 기수가 위쪽으로 들릴 때 +θ의 각을 갖게 되고 (x1, y1, z1)좌표 축은 (x2, y2, z2)이라는 새로운 좌표 축을 갖게 됩니다. 물론 y1축을 기준으로 회전했기 때문에, y1축은 y2축과 동일합니다.

```
| x_2 |   | cosθ   0  -sinθ || x_1 |
| y_2 | = |  0     1    0   || y_1 |
| z_2 |   | sinθ   0   cosθ || z_1 |

B_2 = R_2 B
```
여기서 R2는 Pitch 회전(Rotation) 행렬을 나타내고, B1는 회전이 있기 전의 물체고정좌표계(Body-fixed Frame), B2는 Y1축을 기준으로 회전한 이후의 새로운 물체고정좌표계를 의미합니다.

**3번째 회전은 z2축을 기준으로 한 Yaw 회전**
3번째 회전은 z2축을 기준으로 한 Yaw회전이고, 이를 3번째 오일러 각(3 Euler Angle)이라고 부릅니다.

드론의 기수가 오른쪽으로 돌아갈 때 +ψ의 각을 갖게 되고 (x2, y2, z2) 좌표 축은 (x3, y3, z3)이라는 새로운 좌표 축을 갖게 됩니다. z2축을 기준으로 회전했기 때문에, z2축은 z3축과 동일합니다.

```
| x_3 |   |  cosψ  sinψ   0 || x_2 |
| y_3 | = | -sinψ  cosψ   0 || y_2 |
| z_3 |   |   0     0     1 || z_2 |

B_3 = R_3 B
```

**회전 순서**
물체의 현재의 자세(Orientation)와 관계없이 원하는 어떠한 형태의 자세도 세 번의 오일러 각 회전을 통해서 도달할 수 있다는 것이 핵심입니다.

회전의 순서에 따라 1-2-3 오일러 각/회전, 3-2-1 오일러 각/회전 등으로 부릅니다. 간단하게 매트릭스 연산을 통해서 관계를 나타내면 다음과 같습니다.

```
B3 = R3 B2
B3 = R3 R2 B1
B3 = R3 R2 R1 B
B3 = R123 B
```

매트릭스 회전 행렬이 배치된 순서는 R3R2R1이지만, Roll-Pitch-Yaw순서로 회전했습니다. 정리하자면, 오일러 각/회전은 회전 순서에 따라 1-2-3 오일러 각/회전이라고 부르며, 계산 관계식에서 회전 행렬의 배치는 반대 순서인 R3R2R1가 됩니다.

* 가장 흔하게 쓰는 회전 순서는 3-2-1 오일러 각이라고 합니다.
* 이 경우에는 회전 행렬의 배치가 R1R2R3가 되고, 3-2-1 오일러 각/회전이 됩니다.

**B3에서 B를 구하는 행렬 (오일러 역행렬)**
```
B3 = R3 R2 R1 B
[R3 R2 R1]^-1 B3 = B
R_1^-1 R_2^-1 R_3^-1 B3 = B
```
이와 같이 역행렬을 이용해서 연산을 하면, 거꾸로 B3 프레임을 B프레임으로 변환시키는 행렬을 계산할 수 있습니다. 

각 회전의 역행렬 매트릭스 (R1, R2, R3)는 각 회전 (φ, θ, ψ)의 부호를 반대로 (-φ, -θ, -ψ)함으로써 계산할 수 있습니다.

또한 매트릭스 연산을 통해서 R_1^-1 R_2^-1 R_3^-1 행렬을 계산하게 되면 해당 회전 행렬을 구할 수 있습니다.

* 여기서 역행렬인 회전행렬은 3-2-1 오일러 각이 됩니다.

[**엑셀 시트로 돌려보고 확인하자**](https://docs.google.com/spreadsheets/d/1sGk0MLZbCMCfNXPZ0hJoNsRUQG4K0DlyhorEA1PCNYw/edit#gid=1743987635)

## Euler angle을 이용한 3차원 좌표 변환
Euler angle을 활용하여 두 삼차원 좌표계간의 관계를 표현할 수 있습니다.

**Z-축을 회전 축으로 하여 φ만큼 회전**
P1(1, 0, 0)에 대해여, φ만큼 회전 시킬 경우, 삼각함수 표현법을 통하여, 다음과 같이 변환될 수 있습니다.

```
x = 1 x cosφ   
y = 1 x sinφ
z = 0
```

P2(0, 1, 0)에 대해서 위의 방법을 적용하면,
```
x = 1 x cos(φ + PI / 2) = -sinφ
y = 1 x sin(φ + PI / 2) = cosφ
z = 0
```

P3(0, 0, 1)의 경우는 제자리에 위치합니다.

**점 세 개에 대한 열 기준 행렬로 표현 하여, 축 기준 행렬 값을 얻을 수 있습니다.**
- 정확한 표현인지는 잘 모르겠음.

```
     | cosφ  -sinφ    0  |
Zφ = | sinφ   cosφ    0  |
     |  0      0      1  |

     | cosθ    0   -sinθ |
Yθ = |  0      1      0  |
     | sinθ    0    cosθ |
     
     |  1    0        0  |
Yψ = |  0   cosψ   -sinψ |
     |  0   sinψ    cosψ |
```

로 각각 구할 수 있습니다.

* Euler angle를 사용하는 좌표변환법이 MSE에서 자주 사용됩니다.
    - MSE(Mean square Error, 평균 제곱 오차)라고 하는 것 같은데 잘 모르겠습니다.

**Matrix Orientation 구하기**
자주 사용되는 값을 미리 구합니다.
```
s1 = sinθ1
s2 = sinθ2
s3 = sinθ3
c1 = cosθ1
c2 = cosθ2
c3 = cosθ3
```

후에 Matrix orientation을 구할 수 있습니다.
```
Rot(Y, θ1) Rot(X, θ2) Rot(Z, θ3)
Rot(Z, θ1) Rot(X, θ2) Rot(Z, θ3)
```
## 회전의 종속성

**회전 순서의 영향**
**오일러 각도는 회전 순서를 어떻게 하느냐에 따라 점의 최종 위치가 달라집니다. 오일러 회전은 경로 독립적이지 않습니다. 회전의 순서에 유의해야 합니다.**

아주 작은 회전인 경우, 회전 순서를 바꾸더라도 결과에 영향은 없습니다. 다만 아주 작은 세타가 여러번 곱해지면, 0임을 주의해야 합니다.

[음](https://satlab.tistory.com/91)

## 오일러 각도 변환의 한계
오일러 변환의 한계는 Gimbal lock문제가 있습니다.

오일러 회전은 전역 좌표계의 좌표축을 기준으로 회전합니다.

전역 좌표 계에서 회전이 발생하기 때문에 한 축의 회전이 다른 축의 회전과 겹치는 문제가 발생합니다. 예를 들어, Z축과 평행한 어떤 벡터를 X축을 회전축(pitch)으로 90도 회전해서 그 벡터가 Y축(yaw)과 평행하게 되었다고 하면, 이 때 Y축을 주위로 아무리 회전을 시켜도, Z축을 회전(roll)과 같은 결과가 나옵니다.

즉, 회전을 할 수 있는 축 하나를 일게 되는 것입니다. 이를 짐벌락 이라고 말합니다. (자유도를 잃어버린다 라고도 하는 것 같습니다.)

* **최초의 회전은 그 후의 2번째 3번째 회전축에 영향을 주지 않습니다.**

##  오일러 좌표 (a, b)에서 (c, d)로 가는 측지선의 공식은? 

1. (a, 0)에서 (c, d - b)로 가는 공식을 구한 다음 전체적으로 phi += b를 하면 된다.

2. (a, 0)은 xz 평면에 있는데, xyz 좌표로 보면 (sin a, 0, cos a)일테고...
얘를 북극으로 보내는 변환 행렬은
[[cos a, 0, -sin a],
 [0, 1, 0],
 [sin a, 0, cos a]]
가 된다.

3. (c, d)를 xyz로 변환한 뒤, 앞서 구한 "(a, 0)을 북극으로 보내는 변환 행렬"을 적용한다.
이렇게 얻어진 점을 (c', d')이라 하자.

4. 북극에서 (c', d')으로 가는 공식을 만든다.
(마찬가지로, d' = 0일때 식을 구하고, phi += d'을 하면 더 쉽다.)

5. 공식에 "(a, 0)을 북극으로 보내는 변환 행렬"의 역행렬을 적용한다.

6. 1번에서 말한대로 전체 공식에 phi += b를 하면 된다.

? 전체적으로 phi += b이 무슨뜻인가?

* 축지선이란 쌍곡면에서 두 점을 잇는 곡선 중 거리가 가장 짧은 곡선.

## 오일러 피 함수 (Euler`s phi (totient) function)
오일러 피 함수는 정수론에 등장하는 함수로서 n이하의 자연수 중 n과 서로소인 수의 개수를 구하는 함수입니다. 오일러의 정리와 함께 쓰이기도 하고, 단독으로 사용되기도 합니다. 

**이 글은 독자가 소수를 구하는 알고리즘 중 에라토스테네스의 체를 안다고 가정하고 설명합니다.**

[오일러 피 함수](https://www.weeklyps.com/entry/%EC%98%A4%EC%9D%BC%EB%9F%AC-%ED%94%BC-%ED%95%A8%EC%88%98)

[신소재 공학부 자료인데, 어디다 쓰는 건지 모르겠다아?](https://youngung.github.io/lecturenotes/MetalForming/04_EulerAngles_MF.pdf)
---
layout: post
title: Invoke & InvokeRepeating & Coroutine
---

<details><summary>Invoke & InvokeRepeating</summary>
<div markdown="1">

[MonoBehaviour.Invoke](https://docs.unity3d.com/ScriptReference/MonoBehaviour.Invoke.html)

[MonoBehaviour.InvokeRepeating](https://docs.unity3d.com/ScriptReference/MonoBehaviour.InvokeRepeating.html)

시간 초 단위로 메서드를 호출합니다.

* 메서드에 매개 변수를 전달해야 하는 경우 Coroutine을 대신 사용하는 것이 좋습니다. 코루틴은 또한 더 나은 성능을 제공합니다.

</div></details>

## Coroutine
- [ ] [yield return의 종류](https://yeobi27.tistory.com/entry/Unity-yield-return-종류)

<details><summary>Coroutine</summary>
<div markdown="1">

[Coroutine](https://docs.unity3d.com/ScriptReference/Coroutine.html)

코루틴은 완료 될 때까지 실행(yield)을 일시 중지할 수 있는 함수입니다.

</div></details>

<details><summary>코루틴 예시</summary>
<div markdown="1">

```C#
IEnumerator EffectDestroy()
    {
        Particle.Play();
        while(Particle ? Particle.isPlaying : false)
        {
            float ReverseTime = 1.0f - (Particle.time / Particle.main.duration);
            transform.localScale = new Vector3(ReverseTime, ReverseTime, ReverseTime);
            yield return null;
        }
        Destroy(gameObject);
    }
```

</div></details>
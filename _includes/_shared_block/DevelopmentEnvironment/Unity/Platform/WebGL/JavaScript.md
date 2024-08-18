### JavaScript DLL In Unity WebGL
Unity의 WebGL에서는 java script 함수를 직접 호출하는 방법을 제공합니다. 이를 위해선 `.jslib`확장자를 사용하여 `Assets`폴더의 `Plugins` 하위 폴더에 다음과 같은 JavaScript를 추가합니다. 아래의 코드는 본문의 예시 코드를 축약했습니다.

* JavaScript함수를 호출하기 위해서는 WebGL을 빌드한 다음 실행해야 합니다. 그렇지 않을 경우 함수 호출시 `EntryPointNotFound` 오류가 뜹니다. 

```JavaScript
mergeInto(LibraryManager.library, {

  Hello: function () {
    window.alert("Hello, world!");
  },
  // JavaScript 함수를 추가...
});
```

이제 Unity에서 다음과 같은 방법으로 JavaScript함수를 호출 합니다.
```CSharp
using UnityEngine;
using System.Runtime.InteropServices;

public class NewBehaviourScript : MonoBehaviour {
    [DllImport("__Internal")]
    private static extern void Hello();
    // javascript 함수 호출 추가...
}
```

### HTML Unity 데이터 교환
다음 예시는 URL 파라미터를 Unity에서 읽어오는 예시 코드입니다.

```JavaScript
    GetName: function () {
        // Name Parameter를 리턴합니다.
        var url = new URL(window.location);
        var params = url.searchParams;

        // 예를들어 URL이 다음과 같으면 "http://127.0.0.1:5500/index.html?name=temp" 파라미터 name의 값은 temp가 됩니다.
        var name = params.get("name");

        // console.log("url: " + url);
        // console.log("name: " + name);
        return name;
    }
```

### 참고
* [Unity Documentation, "WebGL: 브라우저 스크립트와 상호작용"](https://docs.unity3d.com/kr/2023.2/Manual/webgl-interactingwithbrowserscripting.html)

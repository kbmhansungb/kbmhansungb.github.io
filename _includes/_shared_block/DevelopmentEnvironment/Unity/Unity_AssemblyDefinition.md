[Assembly definitions](https://docs.unity3d.com/Manual/ScriptCompilationAssemblyDefinitionFiles.html)

Unity에서 어셈블리 정의(Assembly Definition) 파일은 프로젝트의 스크립트를 구조화하고 컴파일 단위를 정의하기 위해 사용되는 파일입니다. 이 파일은 유니티의 스크립트 컴파일러에 의해 처리되며, 스크립트 파일을 논리적으로 그룹화하여 관련된 스크립트 간의 종속성을 관리할 수 있습니다.

어셈블리 정의 파일을 사용하면 다음과 같은 이점을 얻을 수 있습니다:

* 모듈화: 프로젝트의 스크립트를 논리적으로 그룹화하여 모듈화할 수 있습니다. 이는 코드의 구조와 유지 보수를 개선하고, 여러 개발자가 동시에 작업할 때 충돌을 최소화하는 데 도움이 됩니다.
* 컴파일 단위 제어: 어셈블리 정의 파일은 스크립트 파일이 컴파일되는 단위를 정의합니다. 이를 통해 스크립트의 컴파일 순서와 의존성을 명시적으로 관리할 수 있습니다. 또한, 불필요한 스크립트의 재컴파일을 방지하여 빌드 시간을 단축할 수 있습니다.
* 의존성 해결: 어셈블리 정의 파일은 다른 어셈블리에 대한 의존성을 명시적으로 선언할 수 있습니다. 이를 통해 스크립트 간의 종속성을 관리하고, 의존하는 어셈블리의 변경이 있을 때 해당 어셈블리만 다시 컴파일할 수 있습니다.

어셈블리 정의 파일은 Unity Editor에서 생성하고 편집할 수 있으며, 일반적으로 프로젝트의 스크립트 파일과 동일한 위치에 .asmdef 확장자를 가진 파일로 저장됩니다. 이 파일은 JSON 형식으로 작성되며, 스크립트 파일의 소속 그룹, 의존성 등을 지정할 수 있습니다.

정의된 어셈블리는 Unity Editor에서 스크립트 컴파일 시 고려되며, 적절한 컴파일 순서와 종속성 해결을 수행합니다. 이를 통해 프로젝트의 스크립트 구조를 효과적으로 관리할 수 있습니다.

## Untiy Editor Assembly Definition
어셈블리 정의는 빌드할 때 정의에 있는 platforms에 의해 빌드됩니다. 어셈블리를 분리하거나 빌드할 때 포함되지 않도록 Platforms에 Editor만 타겟이 되도록 설정해야 합니다.

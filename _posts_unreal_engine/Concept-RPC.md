---
layout: post
title: RPC
---

# 클라이언트 서버 모델

게임 스테이트에 대해 모든 부분을 결정하는 서버가 하나 있고, 접속된 클라이언트는 가급적 서버에 가깝게 추정합니다.

서버는 중요한 의사결정을 전부 내리고, 모든 권위적 상태가 저장되고, 클라이언트 접속을 처리하고, 새로운 맵으로 이동하며, 경기 시작이나 종료와 같은 전반적인 게임 플레이 흐름을 담당합니다.

## 서버 게임플레이 흐름

서버는 게임플레이 흐름을 담당합니다. (새로운 맵으로 이동할 때가 되었음을, 게임 플레이가 시작 및 종료되었음을, 액터 리플리케이션 업데이트 등과 함께 클라이언트에 알리는 기)

게임플레이 상태와 흐름은 보통 GameMode 액터를 통해 구동됩니다.

서버에만 이 액터의 유효 사본이 저장됩니다.

위의 상태를 클라이언트에게 알리는 용도로 GameState 액터가 있어, GameMode 액터의 중요한 상태를 반영합니다.

이 GameState액터는 각 클라이언트에게 리플리케이트 되도록 마킹되어 있습니다.

클라이언트는 이 GameState 액터에 대한 사본을 추정하여 저장합니다. 이 액터를 참고로 게임의 일반적인 상태를 압니다.

# 접속 프로세스

서버가 네트워킹 관점에서 작업을 하기 위해서는 클라이언트의 접속이 필요합니다. 새 클라이언트가 처음으로 접속할 때, 몇 가지 일들이 벌어집니다.

1. 먼저 클라이언트는 서버에 접속하겠다는 요청을 보냅니다.
2. 서버는 이 요청을 처리하고, 서버가 접속을 거부하지 않으면 적합한 진행 정보를 포함해 클라이언트에게 응답을 보냅니다.

## 접속 단계

1. 클라이언트가 접속 요청을 보냅니다.
2. 서버가 수락하면, 현재 맵을 전송합니다.
3. 서버는 클라이언트가 이 맵을 로드할 때까지 기다립니다.
4. 로드가 되면, 서버 로컬에서 AGameModeBase::PreLogin을 호출합니다.
    - GameMode에 접속을 거부할 수 있는 기회를 줍니다.
5. 수락되면, 서버는 AGameModeBase::Login을 호출합니다.
    - PlayerController를 만들고, 새로 접속된 클라이언트에 리플리케이트 시킵니다.
    - 수신되면 PlayerController가 클라이언트의 접속 과정에서 견본으로 사용되던 임시 PlayerController를 대체합니다.
    - APlayerController::BeginPlay에서 RPC함수를 호출하는 것은 아직 안전하지 않습니다. AGameModeBase::PostLogin호출시까지 기다려야 합니다.
6. 오류가 없을 경우 AGameModeBase::PostLogin이 호출됩니다.
    - 이제 PlayerController에서 RPC함수를 호출해도 안전합니다.

[클라이언트 서버 모델](https://docs.unrealengine.com/4.27/ko/InteractiveExperiences/Networking/Server/)
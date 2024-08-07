[Wiki: communication protocol](https://en.wikipedia.org/wiki/Communication_protocol)

`통신 프로토콜(Communication protocol)`은 통신 시스템에서 두 개 이상의 개체가 정보를 전송하기 위한 규칙 시스템입니다. 이러한 프로토콜은 통신의 규칙, 구문, 의미, 동기화 및 가능한 오류 복구 방법을 정의합니다. 프로토콜은 하드웨어, 소프트웨어 또는 그들의 조합으로 구현될 수 있습니다.

통신 시스템은 잘 정의된 형식을 사용하여 다양한 메시지를 교환합니다. 각 메시지에는 미리 결정된 가능한 응답 범위에서 의미를 명확히 하기 위한 정확한 의미가 있습니다. 이러한 동작은 일반적으로 구현 방법과는 무관하며, 관련 당사자 간의 합의를 통해 결정됩니다.

![image](https://github.com/kbmhansungb/kbmhansungb.github.io/assets/56149613/df63d445-4adf-4127-86c1-0616686a1629)

OSI 모델은 네트워킹의 다양한 기능을 7개의 계층으로 나누어 규정한 것입니다. 각 계층은 특정한 역할을 담당하며, 데이터가 네트워크를 통해 전송될 때 각 계층에서 처리됩니다.

* 물리 계층(Physical Layer)은 데이터 전송의 가장 기초적인 수준을 다룹니다. 여기에는 케이블, 스위치, 네트워크 허브와 같은 하드웨어 장비가 포함됩니다.
* 데이터 링크 계층(Data Link Layer)은 네트워크 상에서 물리적으로 인접한 기기들 사이에서 데이터를 안정적으로 전송하기 위한 프로토콜과 메커니즘을 제공합니다. 여기서는 이더넷과 Wi-Fi와 같은 프로토콜이 작동합니다.
* 네트워크 계층(Network Layer)은 데이터를 여러 네트워크 간에 전달하고, 이를 위한 경로를 설정합니다. IP 주소가 이 계층에서 사용됩니다.
* 전송 계층(Transport Layer)은 호스트 간의 데이터 전송을 관리합니다. TCP와 UDP 같은 프로토콜이 이 계층에서 활용됩니다.
* 세션 계층(Session Layer)은 통신 세션을 설정, 관리, 종료하는 기능을 담당합니다. 이 계층은 통신이 시작되고 유지되며 끝나는 방식을 제어합니다.
* 표현 계층(Presentation Layer)은 데이터의 표현 형식, 즉 인코딩과 디코딩을 다루는 계층입니다. 이 계층은 데이터를 사용자가 이해할 수 있는 형태로 변환하거나, 네트워크를 통해 전송할 수 있는 형태로 변환하는 역할을 합니다.
* 응용 계층(Application Layer)은 사용자에게 가장 가까운 계층으로, 이메일 클라이언트, 웹 브라우저 등 사용자가 직접 사용하는 응용 프로그램이 포함됩니다. 데이터의 최종 목적지로, 실제 사용자의 요청을 서비스로 전환하는 역할을 합니다.

TCP/IP 모델은 네트워킹 프로토콜을 개념적으로 4개의 계층으로 구분한 것으로, 각 계층은 데이터 통신 과정에서 특정 역할을 수행합니다.

* 네트워크 인터페이스 계층(Network Interface Layer)은 이 계층은 네트워크 하드웨어와 드라이버를 포함하여 데이터를 전기적 신호로 변환하고 네트워크를 통해 전송하는 역할을 합니다. OSI 모델의 물리 계층과 데이터 링크 계층에 해당합니다.
* 인터넷 계층(Internet Layer)은 이 계층은 데이터 패킷이 송신자로부터 수신자에게 도달할 수 있도록 경로를 결정하는 기능을 담당합니다. 여기서 IP(Internet Protocol) 주소와 라우팅이 수행됩니다. OSI 모델의 네트워크 계층에 대응합니다.
* 전송 계층(Transport Layer)은 이 계층은 종단 간의 신뢰성 있는 데이터 전송을 책임집니다. TCP(Transmission Control Protocol)는 신뢰성 있는 통신을 보장하는 반면, UDP(User Datagram Protocol)는 보다 빠르지만 신뢰성을 보장하지 않는 방식으로 작동합니다. OSI 모델의 전송 계층과 같은 역할을 합니다.
* 응용 계층(Application Layer)은 사용자와 직접 상호작용하는 소프트웨어 응용 프로그램을 포함합니다. 웹 브라우저, 이메일 클라이언트, FTP(File Transfer Protocol) 클라이언트 등이 여기에 속합니다. OSI 모델의 세션 계층, 표현 계층, 응용 계층이 합쳐진 형태로 볼 수 있습니다.

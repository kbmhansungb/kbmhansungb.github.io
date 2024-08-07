# Comunication
네트워크 통신은 컴퓨터, 서버, 휴대폰 등의 네트워크 장치들이상호간에 데이터를 주고 받는 과저이을 의미합니다. 이 데이터 교환을 가능하게 하는 여러 기술과 프로토콜이 있습니다.

## HTTP
HTTP(Hyper Text Transfer Protocol) 통신 프로토콜은 웹에서 정보를 주고받기 위한 규약 또는 방법입니다. 기본적으로 클라이언트와 서버 간의 데이터 교환을 위해 사용됩니다. 쿨라이언트는 서버에 특정 작업을 요청하는 HTTP 메시지를 보내고, 서버는 이에 대한 응답을 보내는 방식으로 통신합니다.

HTTP 프로토콜은 통신의 효율 및 안정성을 보장하기 위해 개발되었으며, HTTP/1, HTTP/2와 같은 버전들이 있습니다. HTTP/2는 헤더 압축, 한 연결 위의 여러 스트림을 통한 멀티플렉싱 등의 기능을 제공하여 성능을 향상 시킬 수 있습니다.

### HTTP Messages
HTTP 메시지는 서버와 클라이언트 간에 데이터가 교환되는 방식입니다. 메시지에는 서버에서 작업을 트리거하기 위해 클라이언트가 보낸 요청과 서버의 응답 두가지 유형이 있습니다. 이 떄 웹 개발자가 HTTP 메시지를 만드는 일은 거의 없습니다.

#### Request
클라이언트가 서버에 특정 리소스를 요청할 떄 사용하는 메시지로, 예를 들어 요청에 관한 메타데이터를 담고 있는 헤더(Header)와 데이터를 포함할 수 있는 바디(Body)가 있습니다.

#### Response
서버가 클라이언트의 요청에 응답하는 메시지입니다. 메시지에도 대표적으로 헤더와 바디가 있습니다.

### HTTP 예시
HTTP 통신의 예시는 Http 통신을 관리하는 클라이언트를 만들고 네트워크 통신이므로 비동기로 Http 요청을 보내고 서버의 응답을 기다립니다. 응답결과를 Callback으로 받으면 메인 로직과 동기화 합니다. 외부와 통신하는 부분들, 통신을 보내고 받는 부분을 예외처리 합니다.

* .Net 프레임워크에서 HTTP 통신을 위해, HttpClient란 클래스가 있습니다.

## Socket
소켓은 컴퓨터 네트워크의 네트워크 노드의 내에 있는 구조로, 네트워크를 통해 데이터를 송수신 하도록 사용합니다. 인터넷 개발에서 TCP/IP 프로토콜의 표준화로 인해 네트워크 소켓이라고도 사용됩니다.

### TCP
TCP는 신뢰성 있는 데이터 전송을 보장하는 연결 지향적 프로토콜입니다. TCP를 사용하면, 데이터는 정확하게, 순서대로, 오류없이 전송됩니다.

* 연결 설정: 데이터 전송을 시작하기 전에 송수신자 간에 핸드 세이크 과정을 통해 연결이 설정됩니다.
* 데이터 순서 보장: 각 데이터 패킷은 순서 번호가 부여되어, 도착한 패킷을 원래 순서대로 재배열할 수 있습니다.
* 오류 검출 및 수정: 전송 중 패킷 손실이 발생하면, 수신자는 누락된 패킷을 다시 요청할 수 있습니다.

### UDP
UDP는 비연결성 프로토콜로, 데이터 전송의 최소한의 오버헤드를 보장하면서 빠른 속도를 제공합니다. UDP는 신뢰성 보다는 효율성을 중시합니다.

* 비연결성: 연결을 설정하고 유지하는 과정이 없어, 일시적인 데이터 전송에 적합합니다.
* 순서 보장 없음: 패킷들이 도착한 순서대로 처리되지 않고, 순서가 뒤바뀔 수 있습니다.
* 오류 수정 없음: 손실된 패킷에 대한 재전송 요구가 없어, 일부 데이터가 유지될 수 있습니다.

## WebSocket
WebSocket은 실시간 양방향 통신을 가능하게 하는 컴퓨터 통신 프로토콜입니다. 이 프로토콜은 웹 애플리케이션과 서버 간에 지속적인 연결을 유지하며, 저지연 통신을 가능하게 합니다. WebSocket은 서버와 클라이언트가 초기 핸드 세이크를 통해 연결을 맺은 후, 그 연결을 통해 서로에게 데이터를 실시간으로 보내고 받을 수 있습니다.

* 게임, 채팅 어플리케이션, 금융거래 서비스 등 실시간 데이터 교환이 필요한 다양한 애플리케이션에서 널리 사용됩니다.
* WebSocket 통신은 HTTP로 브라우저가 유효한지 확인 한 후, Socket 통신으로 데이터를 주고받습니다.

## 결론
HTTP 메시지는 구조가 간단하고 확장서이 뛰어납니다.

### 참고
* [mdn web docs, "HTTP Messages"](https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages)
* [MSDN, "Make HTTP requests with the HttpClient class"](https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages)
* [WIKI: "Network Socket"](https://en.wikipedia.org/wiki/Network_socket)
* [MSDN, "Socket Class"](https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets.socket?view=net-8.0)
* [MSDN< "WebSocket Class"](https://learn.microsoft.com/en-us/dotnet/api/system.net.websockets?view=netstandard-2.0)

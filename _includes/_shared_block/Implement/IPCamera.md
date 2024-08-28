# IP Camera
IP 카메라(IP Camera)는 인터넷 프로토콜(Internet Protocol)을 사용하여 네트워크를 통해 영상 데이터를 전송하는 카메라를 말합니다. 이 카메라는 아날로그 신호가 아닌 디지털 신호를 사용하며, 데이터를 인터넷이나 로컬 네트워크를 통해 전송할 수 있어 원격지에서 실시간으로 영상을 확인할 수 있는 장점이 있습니다.

IP 카메라는 기존의 아날로그 CCTV 카메라와 달리 별도의 케이블을 설치할 필요 없이 네트워크를 통해 영상을 전송할 수 있어 설치가 상대적으로 간편하고, 고해상도의 영상 데이터를 제공할 수 있다는 장점이 있습니다.

## 통신을 위한 RTSP
* [충청남도 천안시_교통정보 CCTV](https://www.data.go.kr/data/15063717/fileData.do)에서 CCTV 공개 rtsp 주소를 이용하여 테스트 할 수 있습니다. 

RTSP(Real-Time Streaming Protocol)는 인터넷을 통해 실시간으로 오디오와 비디오 데이터를 전송하는 데 사용되는 네트워크 프로토콜입니다. RTSP는 클라이언트가 서버와의 세션을 제어할 수 있게 해 주며, 이를 통해 사용자는 스트리밍 미디어의 재생, 일시 중지, 중지 등의 동작을 제어할 수 있습니다.

RTSP는 기본적으로 스트리밍 데이터를 전송하는 것이 아니라, 클라이언트와 서버 간의 통신을 제어하는 프로토콜입니다. 스트리밍 데이터 자체는 보통 RTP(Real-Time Transport Protocol)와 같은 다른 프로토콜을 통해 전송됩니다.

## RTSP 주소
예를 들어 rtsp 통신을 위한 주소는 `rtsp://210.99.70.120:1935/live/cctv001.stream` 구성요소는 다음과 같습니다:
* `rtsp://:`은 사용 중인 프로토콜이 RTSP(Real-Time Streaming Protocol)임을 나타냅니다. 웹 브라우저에서 `http://`로 시작하는 URL과 비슷한 역할을 합니다.
* `210.99.70.120`은 서버의 IP 주소를 나타냅니다. 이 주소는 클라이언트가 스트리밍 데이터를 요청할 서버의 위치를 나타냅니다. 이 예시에서는 `210.99.70.120`이라는 IP 주소가 사용됩니다.
* `1935`은 포트 번호를 나타냅니다. 포트 번호는 클라이언트가 서버에 연결할 때 사용하는 네트워크 포트를 지정합니다. RTSP는 기본적으로 포트 554를 사용하지만, 여기서는 `1935` 포트가 사용되었습니다. 이는 서버 설정에 따라 달라질 수 있습니다.
* `/live/cctv001.stream`은 서버에서 제공하는 특정 스트림 리소스의 경로를 나타냅니다. `/live/`는 디렉토리 구조를, `cctv001.stream`은 특정 스트림의 이름을 의미합니다. 이 경로를 통해 클라이언트는 어떤 콘텐츠를 요청할지를 서버에 지정하게 됩니다. 이 예시에서는 `cctv001.stream`이라는 스트림이 제공되고 있습니다.

## RTSP ID, PW
ID와 PW를 입력하여 통신해야 하는 경우 `rtsp://{ID}:{PW}@210.99.70.120:1935/live/cctv001.stream`와 같이 rtsp주소에 ID와 PW정보를 포함합니다. HIKVISION 카메라를 예시로 들면 `rtsp://admin:1234567!@192.168.0.152:554/streaming/channels/102​`와 같습니다.

## IP Camera PTZ 제어
* [하이크비전 카메라 PTZ 제어 해결 방법(ONVIF 제외)](https://community.home-assistant.io/t/hikvision-camera-ptz-control-workaround-without-onvif/180366)를 참고하거나 IPCamra 제어 페이지에서 통신을 참고 할 수 있습니다.
* Web IP를 통한 카메라 제어는 주로 ONVIF을 이용합니다. ONVIF는 (Open Network Video Interface Forum)의 약자로 동영상 감시 및 기타 물리적 방범 지역 내의 IP 제품들의 표준 통신 규격입니다. 보안장비끼리 서로 연결되어 동작하게 해주는 프로토콜이라 볼 수 있습니다.
* ONVIF가 아니면서 API가 공개 되지 않은 경우 카메라 회사에 전화 하던지 하여 얻습니다.

IP 카메라에서 PTZ 제어는 카메라의 물리적 움직임과 줌 기능을 원격으로 조작할 수 있는 기능을 의미합니다. PTZ는 Pan, Tilt, Zoom의 약자로, 각각 다음을 의미합니다:
* Pan(팬): 카메라가 수평으로 회전하는 것을 의미합니다. 팬 기능을 통해 카메라를 좌우로 움직여 넓은 수평 범위를 모니터링할 수 있습니다.
* Tilt(틸트): 카메라가 수직으로 회전하는 것을 의미합니다. 틸트 기능을 통해 카메라를 위아래로 움직여, 수직 방향의 다른 각도를 모니터링할 수 있습니다.
* Zoom(줌): 카메라의 렌즈를 확대하거나 축소하는 기능을 의미합니다. 줌 기능을 통해 특정 물체나 영역을 더 자세히 관찰할 수 있습니다.

IP 카메라의 PTZ 제어 기능은 감시 효율성을 크게 높여줄 수 있는 중요한 요소로, 사용자가 원하는 대로 카메라의 시야를 조정하여 더 정확한 모니터링을 할 수 있게 해줍니다.

예를 들어 IP 카메라의 PTZ 제어를 `continuous`모드로 보낸다고 하면 `http://{cameraIP}:{cameraPort}/ISAPI/PTZCtrl/channels/1/continuous`로 http통신을 이용합니다. ID및 PW가 있는 경우 Credentials(자격 증명)에 추가하고 payload에 아래의 정보를 담습니다.
```XML
<?xml version: "1.0" encoding="UTF-8"?>
<PTZData>
    <pan>pan</pan>
    <tilt>tilt</tilt>
    <zoom>zoom</zoom>
</PTZData>
```

# Unity에서 OpenCV를 이용한 IP 카메라 스트리밍
Unity에서 IP카메라 스트리밍은 OpenCV의 VideoCapture를 이용한 예시입니다. 이를 구현하기 위해 `OpenCVForUnity` 패키지를 사용합니다.

## 카메라 추상화
기본 카메라 클래스를 정의합니다. 기본 카메라는 장비와 연결되어 있는지, 렌더 되야하는 마지막 텍스쳐, 평균 연결시간을 구하는 계산 식 등이 구현됩니다.

* Unity에서 Raw Texture는 주로 RawImage 컴포넌트와 함께 사용되며, 텍스처나 비디오와 같은 비트맵 데이터를 UI 요소에 표시할 때 사용됩니다. Unity에서는 Texture 클래스나 그 하위 클래스(Texture2D, RenderTexture 등)를 사용하여 이미지를 다룰 수 있으며, RawImage는 이러한 텍스처를 직접 UI에 렌더링합니다.

```CSharp
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Camera
{
    public delegate void CameraBaseDelegate(CameraBase texture);

    public class CameraBase : MonoBehaviour
    {
        protected object CameraBaseLock = new object();
        
        /// <summary>
        /// <para>카메라 연결 상태입니다.</para>
        /// </summary>
        private bool m_isConnected = false;
        public bool IsConnected { get => m_isConnected; }


        /// <summary>
        /// <para>텍스쳐가 업데이트된 시간들입니다.</para>
        /// <para>평균시간을 알기 위해서는 <see cref="CalculateAverageTime"/>을 이용합니다.</para>
        /// </summary>
        private List<DateTime> m_updateTimes = new List<DateTime>();

        private int m_maxUpdateTimeCount = 20;

        /// <summary>
        /// <para>마지막으로 텍스쳐가 없데이트된 시간입니다.</para>
        /// </summary>
        private DateTime? m_lastUpdatedTime = null;

        /// <summary>
        /// <para>마지막 텍스쳐입니다.</para>
        /// </summary>
        private Texture2D m_lastTexture = null;

        public DateTime? LastReceivedTime { get => m_lastUpdatedTime; }
        public Texture2D LastTexture { get => m_lastTexture; }

        /// <summary>
        /// <para><see cref="LastTexture"/>가 업데이트될 때 호출됩니다.</para>
        /// </summary>
        public CameraBaseDelegate OnUpdateLastTexture = new CameraBaseDelegate((CameraBase _) => { });

        /// <summary>
        /// <para>카메라를 연결합니다.</para>
        /// </summary>
        //[ContextMenu("Connect")] //다형성 적용 안됨
        public virtual void Connect()
        {
            m_isConnected = true;
            m_lastUpdatedTime = DateTime.Now;
        }

        /// <summary>
        /// <para>카메라를 업데이트 합니다.</para>
        /// </summary>
        //[ContextMenu("Update")] //다형성 적용 안됨
        public virtual void Update()
        {

        }

        /// <summary>
        /// <para>카메라 연결을 해제합니다.</para>
        /// </summary>
        //[ContextMenu("Disconnect")] //다형성 적용 안됨
        public virtual void Disconnect()
        {
            m_isConnected = false;
            UpdateLastTexture(null);
        }

        /// <summary>
        /// <para>카메라를 파괴합니다.</para>
        /// </summary>
        protected virtual void OnDestroy()
        {
            Disconnect();
        }

        /// <summary>
        /// <para>마지막 텍스쳐를 업데이트합니다.</para>
        /// <para>카메라 프레임 업데이트 후 호출되야 합니다.</para>
        /// </summary>
        /// <param name="texture"></param>
        protected void UpdateLastTexture(Texture2D texture)
        {
            lock (CameraBaseLock)
            {
                if (m_lastTexture != null)
                {
                    Destroy(m_lastTexture);
                }

                if (texture != null)
                {
                    m_lastTexture = texture;
                    m_lastUpdatedTime = DateTime.Now;

                    // Update Time List
                    m_updateTimes.Add(m_lastUpdatedTime.Value);
                    if (m_updateTimes.Count > m_maxUpdateTimeCount)
                    {
                        m_updateTimes.RemoveAt(0);
                    }
                }
                else
                {
                    m_lastTexture = null;
                    m_lastUpdatedTime = null;

                    m_updateTimes.Clear();
                }
            }
            OnUpdateLastTexture(this);
        }

        /// <summary>
        /// <para>평균 시간을 계산합니다.</para>
        /// </summary>
        /// <returns></returns>
        public float CalculateAverageTime()
        {
            // 연결되어 있지 않은 경우 무한대를 반환합니다.
            if (m_isConnected == false)
            {
                return float.PositiveInfinity;
            }

            if (m_updateTimes.Count >= 2)
            {
                // 마지막 업데이트 시간과 현재 시간의 차이를 계산합니다.
                TimeSpan[] timeSpans = new TimeSpan[m_updateTimes.Count - 1];
                for (int i = 0; i < m_updateTimes.Count - 1; i++)
                {
                    timeSpans[i] = m_updateTimes[i + 1] - m_updateTimes[i];
                }

                // 평균 시간을 계산합니다.
                double totalSeconds = 0.0f;
                for (int i = 0; i < timeSpans.Length; i++)
                {
                    totalSeconds += timeSpans[i].TotalSeconds;
                }

                return (float)(totalSeconds / timeSpans.Length);
            }
            else
            {
                // 일정시간이 지났는데 프레임이 업데이트 되지 않은 경우 무한대를 반환합니다.
                bool isBadConnection = (DateTime.Now - m_lastUpdatedTime.Value).TotalSeconds > 1.0f;
                return isBadConnection ? float.PositiveInfinity : 0.0f;
            }
        }
    }
}
```

## VideoCapture를 이용한 CCTV 카메라
OpenCV의 VideoCaputre를 이용하여 CCTV를 구현합니다. 

VideoCaputre를 Read하여 프레임을 가져옵니다. 이때 프레임을 메인 스레드에서 가져오는 경우 프레임을 가져오면 지연 시간이 크게 발생합니다. 또한 프레임을 가져오는 중에 VideoCapture를 Dispose하면 충돌로 인한 오류(크레쉬)가 발생하여 프로그램이 비정상 종료됩니다. 따라서 충돌을 피하기 위해 Read 주기를 짧게 하여 충돌을 피하거나 Read되는 동안 Dispose가 호출되지 않도로 처리 해야 합니다.

* [Flags for video I/O](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704da6223452891755166a4fd5173ea257068)에서 VideoCapture의 프로퍼티를 get, set하기 위한 ID를 볼 수 있습니다. 

```CSharp
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.VideoioModule;
using System.Collections;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace Camera
{
    /*
     * IP 카메라와 통신을 위한 코드 입니다.
     * 
     * CCTV 연결이 제대로 되어 있는지 확인하기 위한 공공 IP는 아래와 같습니다.
     * rtsp://210.99.70.120:1935/live/cctv001.stream
     */

    /// <summary>
    /// <para>IP 카메라를 표현하는 클래스입니다.</para>
    /// </summary>
    public partial class CCTVCamera : CameraBase
    {
        protected object CCTVCameraLockObject = new object();

        public string URL;

        private Thread m_connectionThread = null;
        private VideoCapture m_videoCapture = null;

        private Coroutine m_readCoroutine = null;

        [Header("CCTV Camera")]
        public float m_updateDelay = 0.0f;

        public override void Connect()
        {
            base.Connect();

            Debug.Log("CCTV Camera is connected.");

            m_connectionThread = new Thread(Thread_Connect);
            m_connectionThread.Start();
        }

        public override void Update()
        {
            base.Update();

            if (m_videoCapture != null && m_videoCapture.isOpened() && IsConnected)
            {
                if (m_readCoroutine == null)
                {
                    m_readCoroutine = StartCoroutine(ReadCoroutine());
                }
            }
        }

        public override void Disconnect()
        {
            base.Disconnect();

            Debug.Log("CCTV Camera is disconnected.");

            if (m_readCoroutine != null)
            {
                StopCoroutine(m_readCoroutine);
                m_readCoroutine = null;
            }

            m_connectionThread?.Abort();
            m_connectionThread = null;

            m_videoCapture?.Dispose();
            m_videoCapture = null;
        }

        private void Thread_Connect()
        {
            // VideoCapture 객체 생성시 자동으로 연결됨
            // VideoCapture가 연결되지 않은 경우, 비정상적으로 시간이 지연될 수 있음
            m_videoCapture = new VideoCapture();

            m_videoCapture.open(URL);
        }

        private IEnumerator ReadCoroutine()
        {
            // FIXME: 속성을 못 찾아서 직접 입력함
            // https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
            const int CAP_PROP_POS_AVI_RATIO = 2;

            Task<Mat> task = Task.Run(() =>
            {
                // frame을 읽을 때까지 대기
                Mat frame = new Mat();
                bool isRead = false;
                while (IsConnected && !isRead)
                {
                    // FIXME: 마지막 프레임을 가져오려고 하면 레잍턴시가 너무 크게 발생함
                    //// 마지막 프레임으로 이동
                    //m_videoCapture.set(CAP_PROP_POS_AVI_RATIO, 1.0);
                    isRead = m_videoCapture.read(frame);
                }
                return frame;
            });


            yield return new WaitUntil(() => task.IsCompleted);

            Mat frame = task.Result;
            Mat rgbaMat = new Mat();

            // BGR -> RGBA로 변환
            Imgproc.cvtColor(frame, rgbaMat, Imgproc.COLOR_BGR2RGBA);
            Texture2D texture = new Texture2D(rgbaMat.cols(), rgbaMat.rows(), TextureFormat.RGBA32, false);

            Utils.matToTexture2D(rgbaMat, texture, true, 0);
            texture.Apply();
            UpdateLastTexture(texture);

            yield return new WaitForSeconds(m_updateDelay); // 통신속도를 테스트하고 싶을때
            m_readCoroutine = null;

        }
    }
}
```

## PTZ 제어 샘플 코드
IP Camera의 PTZ제어를 위한 구현입니다.

```CSharp
using System.IO;
using System.Net;
using System.Text;
using UnityEngine;

namespace Camera
{
    public partial class CCTVCamera : CameraBase
    {
        // CCTV카메라의 PTZ 제어 구현

        private string IP = "192.168.0.151";
        private int Port = 80;  // HTTP 포트는 일반적으로 80입니다
        
        private string ID = "admin";
        private string PW = "1234567!";

        public void ControlPanTiltPTZ(int pan, int tilt)
        {
            string url = $"http://{IP}:{Port}/ISAPI/PTZCtrl/channels/1/continuous";
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
            request.Method = "PUT";

            // PTZ 제어 명령 XML 생성
            string payload = $"<?xml version: \"1.0\" encoding=\"UTF-8\"?><PTZData><pan>{pan}</pan><tilt>{tilt}</tilt></PTZData>";
            byte[] byteArray = Encoding.UTF8.GetBytes(payload);
            request.ContentLength = byteArray.Length;

            Stream dataStream = request.GetRequestStream();
            dataStream.Write(byteArray, 0, byteArray.Length);
            dataStream.Close();

            if (ID != "" && PW != "")
            {
                request.Credentials = new NetworkCredential(ID, PW);
            }
            request.ContentType = "application/xml";

            try
            {
                HttpWebResponse response = (HttpWebResponse)request.GetResponse();
                Stream stream = response.GetResponseStream();
                StreamReader reader = new StreamReader(stream, Encoding.UTF8);
                string result = reader.ReadToEnd();

                reader.Close();
                stream.Close();
                response.Close();

                Debug.Log("PTZ Control Response: " + result);
            }
            catch (WebException ex)
            {
                Debug.LogWarning("PTZ Control Error: " + ex.Message);
            }
        }

        public void ControlZoomPTZ(int zoomVal)
        {
            string url = $"http://{IP}:{Port}/ISAPI/PTZCtrl/channels/1/continuous";
            HttpWebRequest request = (HttpWebRequest)WebRequest.Create(url);
            request.Method = "PUT";

            // PTZ 제어 명령 XML 생성
            string payload = $"<?xml version: \"1.0\" encoding=\"UTF-8\"?><PTZData><zoom>{zoomVal}</zoom></PTZData>";
            byte[] byteArray = Encoding.UTF8.GetBytes(payload);
            request.ContentLength = byteArray.Length;

            Stream dataStream = request.GetRequestStream();
            dataStream.Write(byteArray, 0, byteArray.Length);
            dataStream.Close();

            if (ID != "" && PW != "")
            {
                request.Credentials = new NetworkCredential(ID, PW);
            }
            request.ContentType = "application/xml";

            try
            {
                HttpWebResponse response = (HttpWebResponse)request.GetResponse();
                Stream stream = response.GetResponseStream();
                StreamReader reader = new StreamReader(stream, Encoding.UTF8);
                string result = reader.ReadToEnd();

                reader.Close();
                stream.Close();
                response.Close();

                Debug.Log("PTZ Control Response: " + result);
            }
            catch (WebException ex)
            {
                Debug.LogError("PTZ Control Error: " + ex.Message);
            }
        }
    }
}

```


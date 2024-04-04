## 지도 좌표계
지도 좌표계는 지구 표면의 위치를 수치로 나타내는 시스템입니다. 이러한 좌표계는 지리적 위치를 정확히 지정하기 위해 사용되며, 다양한 방식으로 구성될 수 있습니다. 가장 널리 사용되는 지도 좌표계의 예로는 위도와 경도를 사용하는 지리적 좌표계, 그리고 투영 좌표계가 있습니다. 

## 지리적 좌표계(Geographic Coordinate System, GCS)
- 이 시스템은 지구를 3차원 구체로 가정하고, 위도와 경도를 사용하여 위치를 결정합니다.
- **위도(Latitude):** 적도를 기준으로 북쪽이나 남쪽으로의 각도를 말합니다. 적도는 0도로, 북극은 90도 북위, 남극은 90도 남위로 표시됩니다.
- **경도(Longitude):** 그리니치 천문대를 기준으로 동쪽이나 서쪽으로의 각도를 말합니다. 그리니치 천문대를 지나는 경선은 0도로 설정되며, 동쪽으로 최대 180도, 서쪽으로 최대 180도까지 측정됩니다.
- 이 좌표계는 전 세계 어디에서나 위치를 정확하게 나타낼 수 있지만, 지구의 곡률을 고려해야 한다는 복잡성이 있습니다.

## 투영 좌표계(Projection Coordinate System)
- 지구와 같은 3차원 객체를 2차원 지도나 화면에 나타내기 위해 사용됩니다. 이를 위해 다양한 '투영' 방법이 존재하며, 이는 3차원 지구를 2차원 평면으로 "평탄화"하는 과정을 포함합니다.
- 일반적인 투영 방법에는 원통형, 원추형, 방위형 등이 있으며, 각각의 방법은 지도의 용도나 특정 지역의 지리적 특성에 따라 선택됩니다.
- 투영 방식에 따라 면적, 거리, 방향, 형상 등이 왜곡될 수 있으며, 따라서 어떤 특성을 보존하는 것이 중요한지에 따라 적절한 투영 방식을 선택해야 합니다.

## 좌표계의 종류
### WGS84
- `EPSG:4326, WGS84위경도`: GPS에서 사용하는 위경도 좌표계입니다.
- `EPSG:32652, UTM52N`: UTM 좌표계의 한 영역으로, 위도와 경도를 기반으로 하는 평면 좌표계입니다.
### GRS80
- `EPSG:5179, UTM-K, NHN:2048`: 네이버 지도에서 사용하는 좌표계입니다.
- `EPSG:5181`: 다음지도에서 사용하는 중부원점 좌표계입니다.
### Bessel
- `EPSG:5174`: 2002년 이전의 중부원점, 연속지적도에서 사용하는 좌표계입니다.
- `KATEC`: 네비게이션 등에서 사용되는 좌표계로, 공식 EPSG 코드는 없습니다.

## API
### Naver Static Map
* [Naver COLUD PLATFORM 사용 가이드, "Static Map API"](https://guide.ncloud-docs.com/docs/maps-static-api)
* [Naver API 가이드, "Static Map"](https://api.ncloud-docs.com/docs/ai-naver-mapsstaticmap-raster)
* [Excel - VBA 지도 좌표계 변환 (WGS84, UTM52N, KATEC, UTM-K, 중부원점)](https://egtools.tistory.com/entry/VBAGEOConverter)

## 참고
* [Wikipedia, "Zoom levels"](https://wiki.openstreetmap.org/wiki/Zoom_levels)


# A_AllInOne – UX 기업 분석 All‑in‑One 대시보드

리포지토리 루트의 **`ux_100_dataset.xlsx`** 를 고정 소스로 읽어, 한 페이지에서 A1 ~ A5 탭으로 나눠 동일한 분석 파이프라인을 수행합니다.

## 빠른 시작

```bash
# (Streamlit Cloud가 아닌 로컬 환경)
pip install -r requirements.txt
streamlit run A_AllInOne.py
```

> **Streamlit Cloud 배포 시** `runtime.txt` 로 Python 3.11을 고정하세요. (3.13 환경에서 pandas/openpyxl 설치가 지연되는 사례가 있습니다)

## 파일 구성
```
A_AllInOne.py         # 메인 앱
requirements.txt      # 의존성
runtime.txt           # Python 런타임 고정 (Cloud용)
ux_100_dataset.xlsx   # 분석 대상 엑셀 (repo 루트)
```

## 엑셀 스키마
- **필수**: `Company` (기업명)
- **선택**: `Category` (업종/그룹 라벨)
- **지표**: 나머지 **수치형 컬럼 전체**를 지표로 자동 인식 (예: `UX_Maturity`, `AI_Adoption`, ...)

엑셀은 여러 시트 사용 가능 (`sheet_name=None`). 권장 시트명은 `A1`, `A2`, `A3`, `A4`, `A5` 이며, 앱의 사이드바에서 각 탭이 참조할 시트를 매핑할 수 있습니다.

## 기능
- 1~10 스케일 가정 → **0~100 재스케일 옵션**
- 지표 **가중치(합=1 자동 정규화)** → `CompositeScore` 산출
- **분위(33%/66%) 기반 `Tier`(Low/Mid/Top)**
- **히스토그램 / 카테고리 박스플롯 / 레이더 차트**
- 모든 그래프 하단 **자동 해설(평균/중앙값/IQR/꼬리 방향)**
- 결과 테이블 **CSV 다운로드**

## 배포(Cloud) 체크리스트
1. 루트에 아래 4개 파일이 반드시 존재
   - `A_AllInOne.py`
   - `requirements.txt`
   - `runtime.txt`
   - `ux_100_dataset.xlsx`
2. Streamlit Cloud 설정에서 **Main file path** 를 `A_AllInOne.py` 로 지정
3. 캐시 이슈가 있으면 Cloud에서 **Restart** 또는 **Rerun** / 필요 시 **Clear cache**
4. 앱 사이드바 상단에 Python/패키지 버전이 표시됩니다(디버깅용).

## 트러블슈팅
- **Excel not found**: 루트에 `ux_100_dataset.xlsx` 가 있는지, 정확한 파일명인지 확인
- **Matplotlib import failed**: `requirements.txt`에 `matplotlib` 누락 → 추가 후 재배포
- **로딩이 멈춘다**: Cloud에서 Python 3.13 환경일 가능성 → `runtime.txt` 로 `python-3.11.9` 고정
- **지표 감지 안 됨**: 숫자가 문자열로 저장된 경우 → 엑셀에서 숫자 타입으로 정리

## 라이선스
사내 분석/세미나 용도 전용. 외부 배포 시 데이터 저작권/보안 정책 준수.

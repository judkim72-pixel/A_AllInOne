# AEON UX100 · A-1 · AI Adoption App (Root)

모든 파일을 **GitHub 레포 루트**에 두고 실행하도록 정리했습니다.

## 실행 (로컬/Streamlit Cloud 공통)
```bash
pip install -r requirements.txt
streamlit run aeon_ux100_a1_adoption_app.py
```

## 파일 구성
```
/ (repo root)
├─ aeon_ux100_a1_adoption_app.py   # Streamlit 앱
├─ requirements.txt                # 의존성
└─ README.md
```

## 사용법
- 앱 상단에서 **엑셀 업로드** 또는 서버 경로 입력
- 엑셀의 시트 이름은 **Data**, 필수 컬럼은 **Company**, **AI Adoption Index (0–5)**

## Dataset
- 기본 제공 파일: `ux_100_dataset.xlsx` (레포 루트)
- 앱은 루트의 이 파일이 있으면 **자동 경로**로 채워 넣습니다.
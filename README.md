# DM-Apartment-Price-Prediction

2026-1 데이터마이닝 팀 프로젝트  
서울 아파트 실거래가 예측 - 공공데이터 기반 변수 설계

---

## 시작하기
1. 레포 클론
2. pip install -r requirements.txt
3. .env.example을 .env로 복사 후 카카오 API 키 입력
4. data/raw/, data/processed/, data/final/ 폴더 직접 생성
5. 카카오톡으로 공유받은 데이터 파일을 data/raw/에 넣기

---

## 프로젝트 구조

```
DM-Apartment-Price-Prediction/
├── notebooks/
│   ├── 01_preprocessing.ipynb       # 실거래가 전처리 + 좌표 변환
│   ├── 02_feature_engineering.ipynb # 외부 데이터 병합 + 파생 변수 생성 + EDA
│   └── 03_modeling.ipynb            # 실험 1~5, 모델 학습 및 성능 비교
├── src/
│   └── utils.py                     # 공통 함수 (거리 계산, 카카오 API 등)
├── .env.example                     # API 키 템플릿 (.env 파일 직접 생성 필요)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 데이터 기간

- 분석 기간: **2023년 1월 ~ 2025년 12월 (3년)**
- 가이드 원안(2019~2023, 5년)에서 변경. 변경 사유:
  - 카카오 API 호출량 및 데이터 수집 시간 제약
  - 2023~2025년은 급락 후 회복 구간으로 가격 변동성이 충분히 큼
  - 최신 시점 데이터일수록 외부 시설(지하철역, 학원 등) 위치와의 정합성이 높음

---

## 데이터 파일

데이터파일 - 로컬에서 한번만 다운 후에 data 폴더 및 하위 폴더 생성 후 넣으시면 됩니다.

```
data/
├── raw/        # 원본 데이터
├── processed/  # 전처리된 데이터
└── final/      # 최종 데이터셋
```

---

## 시작하기

### 1. 레포 클론
```bash
git clone https://github.com/팀원아이디/DM-Apartment-Price-Prediction.git
cd DM-Apartment-Price-Prediction
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. API 키 설정
```bash
cp .env.example .env
# .env 파일 열어서 카카오 API 키 입력
```

---

## 실험 구조

| 실험 | 사용 변수 | 목적 |
|------|-----------|------|
| 실험 1 | 물리적 특성 (면적, 층수, 노후도) | 베이스라인 |
| 실험 2 | 실험 1 + 교통 접근성 | 교통 변수 기여도 확인 |
| 실험 3-A | 실험 2 + 교육/생활편의 (긍정) | 긍정 환경 변수 기여도 |
| 실험 3-B | 실험 2 + 혐오시설 (부정) | 부정 환경 변수 기여도 |
| 실험 3-C | 실험 2 + 긍정 + 부정 전체 | 결합 효과 확인 |
| 실험 4 | 전체 변수 | 최종 모델 |
| 실험 5 | 전체 변수, 강남권 vs 비강남권 분리 | 지역별 변수 중요도 비교 |

---

## 사용 모델

- Ridge 회귀
- Random Forest 회귀
- 평가 지표: RMSE, R²

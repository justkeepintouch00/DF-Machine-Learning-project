# DF Machine Learning Project

분류·회귀·비지도학습을 통해 **문제를 지표로 정의하고, 동일 조건에서 대안을 검증하고, 비즈니스 의사결정으로 연결한** 3인 팀 프로젝트입니다.

## 프로젝트 정보

- 과정: DF 「입문자를 위한 머신러닝」
- 기간: 2026.03.03~2026.05.30
- 형태: 3인 팀 프로젝트
- 기여자: 원가형
- 재현 실험: 2026.08.14

공개 저장소는 분류·회귀·비지도학습의 세 영역으로 나눴습니다. 각 노트북은 **팀 프로젝트 당시 원본 셀·주석·출력을 앞부분에 그대로 유지**하고, 맨 아래 `추가 재현 실험 (2026-08-14)` 섹션에 검증 절차를 보완한 실험과 실행 결과를 덧붙였습니다. 별도의 원본 복제 폴더는 두지 않았습니다.

## 바로가기

- [분류 — 당뇨병 위험군 탐지](classification/diabetes_classification.ipynb)
- [회귀 — 주문별 판매금액 예측](regression/sales_prediction.ipynb)
- [비지도학습 — 신용카드 고객 세분화](unsupervised/credit_card_segmentation.ipynb)
- [포트폴리오 최종 실험 보고서](docs/portfolio_report.md)
- [1~7주차 발표자료](weekly-presentations/README.md)

## 핵심 결과

| 분야 | 문제를 지표로 변환 | 최종 선택 | 객관적 결과 |
|---|---|---|---|
| 분류 | 당뇨 환자 누락 최소화 → Recall·F2 | Logistic Regression + LightGBM 이종 앙상블, 검증 임계값 0.425 | RF 기본 대비 FN 6,003→1,116(-81.41%), Recall 0.1508→0.8421, F2 0.1767→0.6022 |
| 회귀 | 미래 주문금액 오차 최소화 → 시간 분할 RMSE | 이상치 유지 XGBoost | Linear 대비 RMSE 5.16%, MAE 3.71% 감소; 이상치 제거 XGBoost보다 RMSE 6.97% 낮음 |
| 비지도 | 실행 가능한 고객군 3개 도출 → 군집 품질·안정성 | log1p + RobustScaler + KMeans(k=3) | Silhouette 0.2455→0.3916(+59.49%), 시드 안정성 평균 ARI 0.9989 |

## 1. 분류 — 당뇨병 위험군 탐지

클래스 불균형 데이터에서는 Accuracy만 높이면 실제 환자를 놓칠 수 있습니다. 따라서 Recall과 Recall에 더 큰 가중치를 주는 F2를 모델 선택 기준으로 설정했습니다.

- 훈련/검증/테스트를 60%/20%/20%로 계층 분할
- 임계값은 테스트가 아니라 검증 데이터의 F2로 선택
- 강사 피드백을 반영해 선형 Logistic Regression과 비선형 LightGBM을 결합
- 다수 클래스 중복 축소 실험도 수행했으나 검증 F2가 더 낮아 미선택

| 지표 | RF 기본 | 최종 앙상블 | 변화 |
|---|---:|---:|---:|
| Accuracy | 0.8655 | 0.6784 | -18.71%p |
| Precision | 0.5643 | 0.2814 | -28.29%p |
| Recall | 0.1508 | 0.8421 | +69.13%p |
| F2 | 0.1767 | 0.6022 | +42.55%p |
| FN | 6,003 | 1,116 | -81.41% |
| FP | 823 | 15,201 | +14,378건 |

Accuracy와 Precision이 낮아지고 FP가 증가했으므로 전반적인 성능 향상이라고 주장하지 않습니다. **검진 누락 비용을 오탐 비용보다 크게 둔 정책에서 FN을 줄인 결과**입니다.

## 2. 회귀 — 주문별 판매금액 예측

강사 피드백은 비선형 트리 모델에서 낮은 선형 상관계수만으로 변수를 제거하지 말고, 이상치를 자동 삭제하지 말며 유지/제거를 비교하라는 것이었습니다.

- 미래 예측 상황을 반영해 날짜순 70%/15%/15% 분할
- 고객 집계 파생변수는 훈련 기간에서만 계산
- 검증·테스트는 유지하고, 훈련 데이터 1,024개 이상치만 제거한 변형과 원본 유지 변형 비교
- 반복 merge에서 생기던 `_x`, `_y` 중복 열 제거

| 모델 | 검증 RMSE | 테스트 RMSE | 테스트 R² |
|---|---:|---:|---:|
| Linear 유지 | 0.9768 | 0.9784 | 0.9684 |
| XGBoost 유지 | 0.9217 | **0.9280** | **0.9715** |
| XGB+LGBM 유지 | **0.9216** | 0.9287 | 0.9715 |
| XGBoost 훈련 이상치 제거 | 0.9993 | 0.9975 | 0.9671 |

앙상블의 검증 RMSE 우위는 0.0108%에 불과하고 이번 재실행에서 측정된 추론시간은 XGBoost보다 약 4.11배 길었습니다. 따라서 `최저 검증 RMSE의 0.1% 이내 후보 중 추론시간 최소` 규칙으로 XGBoost를 선택했습니다.

## 3. 비지도학습 — 신용카드 고객 세분화

비지도학습에서는 정확도가 아니라 군집 품질, 시드 안정성, 규모, 해석 가능성을 평가했습니다.

- 8,950명, 군집 입력 17개 변수
- raw/StandardScaler, log1p/StandardScaler, log1p/RobustScaler, PCA 95% 변형 비교
- 운영 캠페인 3종이라는 요구를 k=3으로 변환
- KMeans k=3에서 log1p+RobustScaler가 가장 높은 Silhouette 기록

| 세그먼트 | 고객 수(비율) | 데이터 특징 | 실행 가설 |
|---|---:|---|---|
| 고구매·쇼핑/VIP 후보군 | 1,307명(14.60%) | 평균 구매액 1,927.21, 구매빈도 0.813 | 프리미엄 혜택·교차판매 |
| 현금서비스 의존 후보군 | 6,403명(71.54%) | 평균 잔액 2,113.43, 현금서비스 1,247.73 | 분할상환 전환·금융건전성 안내 |
| 저활동·재활성화 후보군 | 1,240명(13.85%) | 평균 구매액 339.01, 구매빈도 0.278 | 저비용 재활성화 캠페인 |

위 액션은 데이터에서 도출한 가설입니다. 실제 효과는 캠페인 A/B 테스트의 재구매율, 전환율, 증분 매출로 검증해야 합니다.


## 주차별 발표 기록

매주 발표한 PPT의 PDF 원본을 `weekly-presentations`에 보존했습니다. PDF 내용은 수정하지 않았으며, 파일명 기준으로 1~7주차 순서를 부여했습니다.

| 주차 | 발표자료 | 페이지 | 주요 내용 |
|---:|---|---:|---|
| 1 | [지도학습 데이터·문제 정의](weekly-presentations/week-01_supervised-learning.pdf) | 6 | 분류·회귀 데이터셋 선정과 클래스 불균형 확인 |
| 2 | [지도학습 2회차](weekly-presentations/week-02_supervised-learning.pdf) | 13 | 전처리·모델링 진행 과정 |
| 3 | [지도학습 3회차](weekly-presentations/week-03_supervised-learning.pdf) | 21 | 분류·회귀 실험 확장 |
| 4 | [분류·회귀 프로젝트](weekly-presentations/week-04_classification-regression-project.pdf) | 15 | 문제 정의, 데이터 분석, 모델 비교 |
| 5 | [분류 프로젝트](weekly-presentations/week-05_classification-project.pdf) | 12 | 임계값 조정과 FN/FP 분석 |
| 6 | [회귀 프로젝트](weekly-presentations/week-06_regression-project.pdf) | 12 | 회귀 모델 비교와 성능 개선 과정 |
| 7 | [신용카드 고객 세분화](weekly-presentations/week-07_customer-segmentation.pdf) | 11 | PCA·KMeans·GMM과 비즈니스 세그먼트 |

> 일부 PDF의 내부 주차 표기와 원본 파일명이 일치하지 않는 경우가 있어, 저장소의 주차 번호는 전달받은 파일 순서를 기준으로 정리했습니다.

## 프로젝트 구조

```text
classification/
├── diabetes_classification.ipynb
└── results/                 # 모델 비교표와 최종 요약
regression/
├── sales_prediction.ipynb
└── results/                 # 이상치 유지/제거 비교와 최종 요약
unsupervised/
├── credit_card_segmentation.ipynb
└── results/                 # 군집 후보·프로파일·비즈니스 가설
weekly-presentations/        # 1~7주차 발표 PDF 원본
docs/
└── portfolio_report.md
DATA_SOURCES.md
requirements.txt
```

## 원가형 기여 내용

- 분류: 초기 데이터·발표자료, 불균형 문제 정의, FN/FP 분석과 결과 정리
- 회귀: 전처리·상관관계 분석, log1p/Box-Cox·잔차 검토, Ridge/Lasso/CV, 이상치 유지/제거 재현
- 비지도: KMeans/GMM 비교, 비즈니스 세그먼트·발표자료, PCA 입력의 타깃 누수 발견·수정
- 공통: 발표·코드 리뷰, 최종 재현 구조와 성능 근거 정리

공동 Colab 작업을 사후 GitHub로 이전했으므로 최초 Git 커밋 작성자가 과거 코드 전체의 단독 작성자를 뜻하지 않습니다. 정확한 셀별 작성자를 확인할 수 없는 부분은 팀 공동 작업으로 표시합니다.

## 재현 방법

```bash
python -m pip install -r requirements.txt
```

저장소 루트에서 Jupyter를 실행한 뒤 각 영역의 노트북을 열어 `Restart & Run All`을 실행합니다. 공개 Kaggle 데이터는 KaggleHub를 통해 자동으로 다운로드됩니다.

## 한계

- 분류 운영 임계값은 실제 FN/FP 비용과 검진 가능 인원이 정해지면 다시 선택해야 합니다.
- 회귀 데이터는 합성 데이터이므로 실제 거래 데이터에서 외부 검증이 필요합니다.
- 고객 세그먼트의 비즈니스 효과는 현재 데이터에 없으며 A/B 테스트가 필요합니다.
- 의료 데이터 분석 결과는 진단 도구가 아니며 인과관계로 해석할 수 없습니다.

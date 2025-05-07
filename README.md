🚗 중고차 가격 예측 모델링 프로젝트
🧠 프로젝트 개요
본 프로젝트는 다양한 차량 정보를 활용하여 중고차의 적정 가격을 예측하는 머신러닝 모델을 구축하는 것을 목표로 합니다. 이를 통해 소비자와 판매자 모두에게 유의미한 가치를 제공하고자 합니다.

🎯 주제 선정 이유
대한민국을 포함한 여러 국가에서 자동차 보급률이 높아지고 있으며, 중고차 시장은 신차 시장 이상으로 활발히 작동 중입니다.

중고차 가격은 차량의 연식, 주행거리, 연료 타입, 브랜드, 변속기 종류 등 다양한 요소에 따라 복잡하게 결정됩니다.

이를 정량적으로 예측할 수 있는 시스템은 소비자와 판매자 모두에게 유의미한 가치를 제공합니다.

🔍 데이터 수집 및 전처리
데이터 출처: Kaggle 데이터셋

데이터 크기: 7,253행 → 정제 후 6,017행

주요 컬럼: 차 이름, 제조연도, 주행거리, 연료타입, 변속기, 소유자 수, 연비, 배기량, 출력, 좌석 수, 신차 가격, 중고차 가격

전처리 작업:

브랜드 추출, 연식 계산(현재연도 - 제조연도), 차량 크기 구분(경차/소형/중형/대형) 등 새로운 특성 파생

범주형 변수는 One-Hot Encoding

수치형 변수는 Z-score 정규화

이상치 제거 및 결측치 보정 수행
ScienceON
+3
GitHub
+3
크몽
+3

🛠 사용 기술 및 라이브러리
데이터 처리 및 시각화:

Pandas, NumPy, Matplotlib, Seaborn, Missingno, koreanize_matplotlib

전처리 및 통계 분석:

Scipy, StandardScaler, OneHotEncoder, PolynomialFeatures

모델링 및 평가:

LinearRegression, train_test_split, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

📁 프로젝트 구조
<pre>
'''
project/
├── data/
│   └── used_car_data.csv
├── notebooks/
│   └── car_price_prediction.ipynb
├── images/
│   └── price_distribution.png
└── README.md
'''
</pre>
📈 주요 결과
다양한 회귀 모델을 비교한 결과, 선형 회귀 모델이 중고차 가격 예측에 적합한 성능을 보였습니다.

모델의 성능은 R² 점수와 RMSE를 기준으로 평가하였으며, 예측 정확도가 높게 나타났습니다.

특히, 차량의 연식과 주행거리가 중고차 가격에 큰 영향을 미치는 변수로 확인되었습니다.

📌 결론 및 시사점
중고차 가격 예측 모델은 소비자에게는 합리적인 구매 결정을, 판매자에게는 적절한 가격 책정을 위한 도구로 활용될 수 있습니다.

향후에는 더 다양한 변수와 고급 모델을 활용하여 예측 정확도를 높일 수 있습니다

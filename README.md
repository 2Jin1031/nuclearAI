# nuclearAI

## 9가지 성능 평가
model_LogisticRegression()_dataset_case1: 0.929232995658466, 0.21135136223765713
model_LogisticRegression()_dataset_case2: 0.7028943560057888, 0.9340554119243807
model_LogisticRegression()_dataset_case3: 0.935383502170767, 0.2068400944605147
model_RandomForestClassifier()_dataset_case1: 1.0, 0.0005971774031516458
model_RandomForestClassifier()_dataset_case2: 1.0, 0.0005182288823305014
model_GradientBoostingClassifier()_dataset_case1: 0.9999276410998553, 0.00016161167151968115
model_GradientBoostingClassifier()_dataset_case2: 1.0, 0.00048277193349402123
model_GradientBoostingClassifier()_dataset_case3: 1.0, 1.4470433593717277e-06
model_LogisticRegression()_dataset_case3: 0.935383502170767, 0.2068400944605147

## step7 실행 방법
input_transform 함수의 4개 변수 케이스별로 지정

update 함수의 로드할 모델에서 앞서 선택한 케이스에서 3개 모델중 하나를 지정

##################### 모델 학습
import pandas as pd
from joblib import dump

def train_eval_by_9cases(model, dataset_case):
    dataset = pd.read_csv(f'./Finals/setting/files_by_cases/dataset_case{dataset_case}.csv')
    dataset.columns = dataset.columns.astype(str)

    from sklearn.model_selection import train_test_split

    x_train = dataset.iloc[:, 2:]
    y_train = dataset.iloc[:, 1]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=2025)
    model.fit(x_train, y_train)

    dump(model, f'./Finals/setting/files_by_cases/model_{str(model)}_dataset_case{dataset_case}.joblib')

    from sklearn.metrics import accuracy_score, log_loss #정확도, 크로스엔트로피
    print(f'model_{str(model)}_dataset_case{dataset_case}: {accuracy_score(y_test, model.predict(x_test))}, {log_loss(y_test, model.predict_proba(x_test))}')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()



# Gradientboosting + selecting variables + pca
train_eval_by_9cases(gbc, 3)


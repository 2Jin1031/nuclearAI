##################### 모델 학습
import pandas as pd
from joblib import dump

dataset = pd.read_csv('dataset.csv')
dataset.columns = dataset.columns.astype(str)

from sklearn.model_selection import train_test_split

x_train = dataset.iloc[:, 2:]
y_train = dataset.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=2025)

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
print('1')
model.fit(x_train, y_train)
print('2')
dump(model, 'model.joblib')

y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test, y_pred), classification_report(y_test, y_pred))

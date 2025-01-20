import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

############## 모든 파일 하나의 데이터셋으로
import os

label = ['normal', 'loca', 'sgtr', 'mslb_inside', 'mslb_outside']


def combine_data(class_num):
    file_path = 'data/' + label[class_num]
    file_list = os.listdir(file_path)
    dataset = pd.DataFrame()
    for file in file_list:
        data = pd.read_csv(os.path.join(file_path, file))
        data['label'] = class_num
        dataset = pd.concat([dataset, data])

    return dataset


dataset = pd.DataFrame()
for i in range(5):
    data = combine_data(i)
    dataset = pd.concat([dataset, data], ignore_index=True)

############# 나머지 변수 pca로 추출
from sklearn.decomposition import PCA
from joblib import dump

selected_cols = ['label', 'KCNTOMS', 'ZINST22', 'UCTMT', 'WSTM1', 'WSTM2', 'WSTM3', 'ZINST102']
need_dataset = dataset[selected_cols]
etc_dataset = dataset.drop(columns=selected_cols)

# PCA 모델 학습
pca = PCA(n_components=10)
etc_dataset = pca.fit_transform(etc_dataset)

# PCA 모델 저장
dump(pca, 'pca_transformer.joblib')

etc_index = [f'extra value {i}' for i in range(10)]
etc_dataset = pd.DataFrame(data=etc_dataset, columns=etc_index)

dataset = pd.concat([need_dataset, etc_dataset], axis=1)

################## MinMax Normalization
import os
import pandas as pd


def minmax(y, y_min, y_max):
    if y_min == y_max:
        return 0
    else:
        return (y - y_min) / (y_max - y_min)


numeric_dataset = dataset.drop(columns=['KCNTOMS', 'label'])
para_list = list(numeric_dataset)

# min, max 값 구해줍니다 =============================================================================
para_min = [dataset[para].min() for para in para_list]
para_max = [dataset[para].max() for para in para_list]
# print(para_min)
# print(para_max)

for i, para in enumerate(para_list):
    dataset[para] = minmax(dataset[para], para_min[i], para_max[i])

# min, max 값은 추후 사용을 위해 저장해줍시다 ==========================================================
with open('minmax.csv', 'w') as f:
    f.write(','.join(para_list) + '\n')
    # 두 번째 줄에 para_min 값을 작성
    f.write(','.join(map(str, para_min)) + '\n')
    # 세 번째 줄에 para_max 값을 작성
    f.write(','.join(map(str, para_max)) + '\n')

dataset.to_csv('dataset.csv')

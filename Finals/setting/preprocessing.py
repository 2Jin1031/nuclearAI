import pandas as pd
import os
from sklearn.decomposition import PCA

class make_dataset:
    def __init__(self, selected_varialbe : list, do_pca : bool):
        self.seleted_variable = selected_varialbe
        self.do_pca = do_pca 
        self.label = ['normal', 'loca', 'sgtr', 'mslb_inside', 'mslb_outside']


    def combine_data(self, class_num):
        file_path = '../../data/' + self.label[class_num]
        file_list = os.listdir(file_path)
        dataset = pd.DataFrame()
        for file in file_list:
            data = pd.read_csv(os.path.join(file_path, file), encoding='utf-8')
            data['label'] = class_num
            dataset = pd.concat([dataset, data])

        return dataset


    def minmax(self, y, y_min, y_max):
        if y_min == y_max:
            return 0
        else:
            return (y - y_min) / (y_max - y_min)


    def make_dataset(self):
        dataset = pd.DataFrame()
        for i in range(5):
            data = self.combine_data(i)
            dataset = pd.concat([dataset, data], ignore_index=True)

        ############# 나머지 변수 pca로 추출

        selected_cols = ['label', 'KCNTOMS'] + self.seleted_variable
        need_dataset = dataset[selected_cols]
        etc_dataset = dataset.drop(columns=selected_cols)

        # PCA 모델 학습

        self.pca = PCA(n_components=10)
        etc_dataset = self.pca.fit_transform(etc_dataset)


        etc_index = [f'extra value {i}' for i in range(10)]
        etc_dataset = pd.DataFrame(data=etc_dataset, columns=etc_index) if self.do_pca else pd.DataFrame()

        dataset = pd.concat([need_dataset, etc_dataset], axis=1)

        ################## MinMax Normalization
        
        numeric_dataset = dataset.drop(columns=['KCNTOMS', 'label'])
        self.para_list = list(numeric_dataset)

        # min, max 값 구해줍니다 =============================================================================
        self.para_min = [dataset[para].min() for para in self.para_list]
        self.para_max = [dataset[para].max() for para in self.para_list]

        for i, para in enumerate(self.para_list):
            dataset[para] = self.minmax(dataset[para], self.para_min[i], self.para_max[i])

        return dataset
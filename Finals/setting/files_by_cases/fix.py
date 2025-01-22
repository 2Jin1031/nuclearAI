import pandas as pd

df= pd.read_csv('Finals/setting/files_by_cases/dataset_case3.csv')

for i in range(len(df)):
    if (df.loc[i, 'KCNTOMS'] == 150) & (df.loc[i, 'label'] == 2):
        df.drop([i], axis=0, inplace=True) 

df.to_csv('Finals/setting/files_by_cases/dataset_case3_fix.csv')
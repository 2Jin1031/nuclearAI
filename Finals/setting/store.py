from joblib import dump
from preprocessing import make_dataset

def store(selected_variable:list, do_pca:bool, case:int):
    d = make_dataset(selected_variable, do_pca)
    dataset = d.make_dataset()


    # PCA 모델 저장
    dump(d.pca, f'./Finals/setting/files_by_cases/pca_transformer_case{case}.joblib')

    # min, max 값은 추후 사용을 위해 저장해줍시다 ==========================================================
    with open(f'./Finals/setting/files_by_cases/minmax_case{case}.csv', 'w') as f:
        f.write(','.join(d.para_list) + '\n')
        # 두 번째 줄에 para_min 값을 작성
        f.write(','.join(map(str, d.para_min)) + '\n')
        # 세 번째 줄에 para_max 값을 작성
        f.write(','.join(map(str, d.para_max)) + '\n')

    dataset.to_csv(f'./Finals/setting//files_by_cases/dataset_case{case}_fix.csv')


# case3 (selecting variables + pca)
store(['ZINST22', 'UCTMT', 'WSTM1', 'WSTM2', 'WSTM3', 'ZINST102'], True, 3)


import pandas as pd

encodings = ['utf-8', 'latin1', 'cp949', 'ISO-8859-1']

for encoding in encodings:
    try:
        data = pd.read_csv("./data/loca/loca_loop1_100_cold 오후 2.01.30.csv", encoding=encoding)
        print(f"성공적으로 읽음: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"인코딩 실패: {encoding}")
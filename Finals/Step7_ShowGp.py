from CNSenv import CommunicatorCNS
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import TextBox, Widget

import joblib
import pandas as pd
import numpy as np
import torch
from torch.nn.functional import softmax

if __name__ == "__main__":

    def minmax(y, y_min, y_max):
        if y_min == y_max:
            return 0
        else:
            return (y - y_min) / (y_max - y_min)


    def input_transform(data):
        # PCA 수행
        selected_cols = ['KCNTOMS', 'ZINST22', 'UCTMT', 'WSTM1', 'WSTM2', 'WSTM3', 'ZINST102']
        need_data = data[selected_cols]
        # etc_data = data.drop(columns=selected_cols)
        #
        # pca = joblib.load('pca_transformer.joblib')
        # etc_data_arr = pca.transform(etc_data)
        #
        # etc_index = [f'extra value {i}' for i in range(10)]
        # etc_data = pd.DataFrame(columns=etc_index)
        # etc_data.loc[0] = pd.Series(data=etc_data_arr.flatten()).values
        #
        # data = pd.concat([need_data, etc_data], axis=1)

        data = need_data

        # minmax 수행
        minmax_db = pd.read_csv('minmax.csv')

        para_list = list(minmax_db.columns)
        para_min = list(minmax_db.iloc[0])
        para_max = list(minmax_db.iloc[1])

        for i, para in enumerate(para_list):
            data[para] = minmax(data[para], para_min[i], para_max[i])

        return data


    cnsenv = CommunicatorCNS(com_ip='172.30.1.75', com_port=7132)
    para_list = ['KCNTOMS', 'KBCDO23', 'ZINST58']
    para_min = [0, 7, 101.3265609741211]
    para_max = [52, 99, 156.2834014892578]

    # 데이터 저장 변수
    sim_time = [0]
    net_out_NORMAL = [0]
    net_out_LOCA = [0]
    net_out_SGTR = [0]
    net_out_MSLB_inside = [0]
    net_out_MSLB_outside = [0]
    NetDiag = ''

    # 그래프 초기화

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_position([0.10, 0.30, 0.85, 0.65])

    line_NORMAL, = ax.plot(sim_time, net_out_NORMAL, label='NORMAL', color='black', linewidth=2.5)
    line_LOCA, = ax.plot(sim_time, net_out_LOCA, label='LOCA', color='blue', linewidth=2.5)
    line_SGTR, = ax.plot(sim_time, net_out_SGTR, label='SGTR', color='red', linewidth=2.5)
    line_MSLB_inside, = ax.plot(sim_time, net_out_MSLB_inside, label='MSLB_in', color='green', linewidth=2.5)
    line_MSLB_outside, = ax.plot(sim_time, net_out_MSLB_outside, label='MSLB_out', color='magenta', linewidth=2.5)

    Label_Dig, = fig.text(0.1, 0.15, f'Diagnosis : {NetDiag}', wrap=True),
    Label_NORMAL, = fig.text(0.1, 0.10, f'NORMAL : {net_out_NORMAL[-1]}', wrap=True, backgroundcolor='#eefade'),
    Label_LOCA, = fig.text(0.1, 0.05, f'LOCA : {net_out_LOCA[-1]}', wrap=True, backgroundcolor='#eefade'),
    Label_SGTR, = fig.text(0.1, 0.01, f'SGTR : {net_out_SGTR[-1]}', wrap=True, backgroundcolor='#eefade'),
    Label_MSLB_inside, = fig.text(0.5, 0.10, f'MSLB inside : {net_out_MSLB_inside[-1]}', wrap=True,
                                  backgroundcolor='#eefade'),
    Label_MSLB_outside, = fig.text(0.5, 0.05, f'MSLB outside : {net_out_MSLB_outside[-1]}', wrap=True,
                                   backgroundcolor='#eefade'),

    ax.set_title("AI Result")
    ax.set_xlabel("Time (sim. tick)")
    ax.set_ylabel("SoftMax Out")
    ax.axhline(y=0.9, linestyle=':', color='black')
    ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1))

    with open('GP_Result_team.csv', 'w') as f:
        f.write(f'Tick,NORMAL,LOCA,SGTR,MSLB_in,MSLB_out,Diagnosis,Diagnosis_label\n')


    def update(frame):

        # CNS 데이터 취득
        is_updated = cnsenv.read_data()

        if is_updated:
            # 실시간 데이터 읽기
            all_para = cnsenv.mem.keys()
            Vals = [cnsenv.mem[para]['Val'] for para in all_para]

            input = pd.DataFrame(columns=all_para)
            input.loc[0] = Vals

            input = input_transform(input)  # 학습에서 사용했던 방식으로 입력 처리

            model = joblib.load('model.joblib')
            out = model.predict(input)
            soft_out = model.predict_proba(input)

            # 그래프 데이터 저장
            sim_time.append(cnsenv.mem['KCNTOMS']['Val'])
            net_out_NORMAL.append(soft_out[0])
            net_out_LOCA.append(soft_out[1])
            net_out_SGTR.append(soft_out[2])
            net_out_MSLB_inside.append(soft_out[3])
            net_out_MSLB_outside.append(soft_out[4])

            NetDiag = {0: "Normal", 1: "LOCA", 2: "SGTR", 3: "MSLB inside", 4: "MSLB outside"}[out]

            with open('GP_Result_team.csv', 'a') as f:
                f.write(
                    f'{cnsenv.mem["KCNTOMS"]["Val"]},{net_out_NORMAL[-1]},{net_out_LOCA[-1]},{net_out_SGTR[-1]},{net_out_MSLB_inside[-1]},{net_out_MSLB_outside[-1]},{NetDiag},{out}\n')

            # 그래프 업데이트
            line_NORMAL.set_data(sim_time, net_out_NORMAL)
            line_LOCA.set_data(sim_time, net_out_LOCA)
            line_SGTR.set_data(sim_time, net_out_SGTR)
            line_MSLB_inside.set_data(sim_time, net_out_MSLB_inside)
            line_MSLB_outside.set_data(sim_time, net_out_MSLB_outside)

            # 그래프 라벨 업데이트
            Label_Dig.set_text(f'Diagnosis : {NetDiag}')
            Label_NORMAL.set_text(f'NORMAL : {net_out_NORMAL[-1] * 100:.2f}[%]')
            Label_LOCA.set_text(f'LOCA : {net_out_LOCA[-1] * 100:.2f}[%]')
            Label_SGTR.set_text(f'SGTR : {net_out_SGTR[-1] * 100:.2f}[%]')
            Label_MSLB_inside.set_text(f'MSLB inside : {net_out_MSLB_inside[-1] * 100:.2f}[%]')
            Label_MSLB_outside.set_text(f'MSLB outside : {net_out_MSLB_outside[-1] * 100:.2f}[%]')

            ax.set(xlim=(0, sim_time[-1] + 1), ylim=(-0.05, 1.05))


    # 애니메이션 시작
    ani = FuncAnimation(fig, update, interval=60, cache_frame_data=False)
    plt.show()

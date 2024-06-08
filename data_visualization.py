import numpy as np
import matplotlib.pyplot as plt
from train_lightning import IndentDataset



def main(fpath):
    datasets_list = []
    BxyF_list = []
    for raw_data_path in fpath:
        datasets_list.append(IndentDataset(raw_data_path, skip=0))
        print("Loaded dataset length: ",len(datasets_list[-1]))
        BxyF = np.concatenate([datasets_list[-1].B, datasets_list[-1].xyF], axis=1)
        BxyF_list.append(BxyF)

    for idx in range(0,390):
        plt.clf()
        for i in range(len(BxyF_list)):
            plt.plot(np.arange(0,15), BxyF_list[i][idx, 0:15])
        plt.grid()
        plt.ylim([-15,15])
        plt.pause(0.1)

    plt.show()

if __name__ == '__main__':
    fpath = [
            # './combined_data_20240605_1414.csv',
            # './combined_data_20240605_1436.csv', 
            # './combined_data_20240605_1458.csv', 
            # # './combined_data_20240605_1531.csv',
            # './combined_data_20240605_1604.csv',
            # './combined_data_20240603_1603.csv', 
            # './combined_data_20240603_1709.csv', 
            # './combined_data_20240603_1625.csv', 
            # './combined_data_20240521_1948.csv',
            # './combined_data_20240603_1648.csv',
            # 'NC_notbrian_combined_20240603_170947.csv',
            'R_notbrian_combined_20240607_201327.csv',
            # 'ZR_notbrian_combined_20240607_223438.csv',
            # 'notbrian_combined_20240603_170947.csv'
            ]
    main(fpath)
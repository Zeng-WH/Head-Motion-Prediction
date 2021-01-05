import preprocess
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset

def main():
    data_set = preprocess.read_all_file_new('E:\模式识别2020\头部姿态预测\数据')
    data_pre = preprocess.Prepross_Data(data_set, 20, 50)
    x_set, y_set = data_pre.data_split()
    seed_1 = np.arange(len(x_set))
    # print(seed_1)
    np.random.shuffle(seed_1)
    x_train_set = x_set[seed_1[0: math.floor(0.8 * len(x_set))]]
    y_train_set = y_set[seed_1[0: math.floor(0.8 * len(x_set))]]
    x_val_set = x_set[seed_1[math.floor(0.8 * len(x_set)): len(x_set)]]
    y_val_set = y_set[seed_1[math.floor(0.8 * len(x_set)): len(x_set)]]
    np.save('x_train_set.npy', x_train_set)
    np.save('y_train_set.npy', y_train_set)
    np.save('x_val_set.npy', x_val_set)
    np.save('y_val_set.npy', y_val_set)


if __name__ == '__main__':
    main()

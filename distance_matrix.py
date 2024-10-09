import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['R_HOME'] = 'D:/R-4.4.0'

from utils import read_GRN

def GRN_hotfig(GRN):
    # 获得GRN的in和out
    GRN_copy = np.copy(GRN)
    dim_in = np.sum(np.abs(GRN_copy), axis = 1)
    dim_out = np.sum(np.abs(GRN_copy), axis = 0)
    return dim_in, dim_out
def caculate_distance(GRNs):
    cellNumber = len(GRNs)
    transition_proba = np.zeros((cellNumber, cellNumber))
    for i in range(cellNumber):
        dim_in1,dim_out1 = GRN_hotfig(GRNs[i])
        dim_in1, dim_out1 = dim_in1.reshape(-1, 1), dim_out1.reshape(-1, 1)
        arrayi = np.concatenate((dim_in1, dim_out1), axis=1)
        for j in range(cellNumber):
            dim_in2,dim_out2 = GRN_hotfig(GRNs[j])
            dim_in2, dim_out2 = dim_in2.reshape(-1, 1), dim_out2.reshape(-1, 1)
            arrayj = np.concatenate((dim_in2, dim_out2), axis=1)
            sim = np.linalg.norm(arrayi - arrayj)
            transition_proba[i, j] = sim
    return transition_proba


def save_distance(save_path):
    pooled_data = pd.read_csv(save_path+'/pooled_data.csv', index_col=0)
    hvg_express_array = pooled_data.values
    GRNs= read_GRN(hvg_express_array, save_path)
    transition_proba = caculate_distance(GRNs)
    np.savetxt(save_path+'/distance.txt', transition_proba)

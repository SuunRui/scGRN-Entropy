import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['R_HOME'] = 'D:/R-4.4.0'
import scanpy as sc


def GRN_hotfig(GRN):
    # 获得GRN的in和out
    GRN_copy = np.copy(GRN)
    dim_in = np.sum(np.abs(GRN_copy), axis = 1)
    dim_out = np.sum(np.abs(GRN_copy), axis = 0)
    return dim_in, dim_out

def caculate_transition_proba(GRNs):
    transition_proba = np.empty((len(GRNs), len(GRNs)))
    for i in range(len(GRNs)):
        dim_in1,dim_out1 = GRN_hotfig(GRNs[i])
        dim_in1, dim_out1 = dim_in1.reshape(-1, 1), dim_out1.reshape(-1, 1)
        arrayi = np.concatenate((dim_in1, dim_out1), axis=1)
        for j in range(len(GRNs)):
            dim_in2,dim_out2 = GRN_hotfig(GRNs[j])
            dim_in2, dim_out2 = dim_in2.reshape(-1, 1), dim_out2.reshape(-1, 1)
            arrayj = np.concatenate((dim_in2, dim_out2), axis=1)
            transition_proba[i, j] = np.linalg.norm(arrayi - arrayj)
        transition_proba[i, :] = np.max(transition_proba[i, :]) - transition_proba[i, :]
        transition_proba[i, i] = 0
        transition_proba[i, :] = transition_proba[i, :] / np.sum(transition_proba[i, :])
    return transition_proba







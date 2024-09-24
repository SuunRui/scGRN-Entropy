import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['R_HOME'] = 'D:/R-4.4.0'
import scanpy as sc
import scanpy as sc
import anndata as ad
import matplotlib.colors as mcolors
import rpy2.robjects as robjects
from scipy.optimize import minimize
import networkx as nx
from pathlib import Path
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr

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

# def accumulate_transition_proba(transition_proba):
#     eigenvalues, eigenvectors = np.linalg.eig(transition_proba)
#     # 找到单位特征值对应的索引
#     stationary_index = np.where(np.isclose(eigenvalues, max(eigenvalues)))[0]
#     transition_probaN = transition_proba
#     for item in stationary_index:
#     # 提取对应的单位特征向量作为平稳分布
#         stationary_distribution = np.real(eigenvectors[:, item])
#         transition_probaN = transition_probaN - np.dot(stationary_distribution.reshape(-1, 1), stationary_distribution.reshape(1, -1))
#     accumulate_transition_probaN = np.linalg.inv(np.eye(len(transition_probaN)) - transition_probaN) - np.eye(len(transition_probaN))
#     accumulate_transition_probaN[accumulate_transition_probaN<0]=0
#     return stationary_index, transition_probaN, accumulate_transition_probaN

def sc_entry(p):
    # 计算单细胞的熵
    entry_list = []
    for item in p:
        if item == 0:
            entry_list.append(0)
        else:
            entry_list.append(-item*np.log(item))
    return np.sum(entry_list)

# def accumulate_transition_proba2(accumulate_transition_probaN):
#     cell_entry = []
#     for item in accumulate_transition_probaN:
#         cell_entry.append(sc_entry(item))
#     cell_entry = np.array(cell_entry)
#     cell_entry_compar = np.zeros((len(cell_entry), len(cell_entry)))
#     for i in range(len(cell_entry)):
#         for j in range(len(cell_entry)):
#             if cell_entry[i] > cell_entry[j]:
#                 # 如果细胞i的熵大于细胞j的熵，则可能发生转移
#                 cell_entry_compar[i,j] = 1
#     accumulate_transition_probaN[np.where(cell_entry_compar == 0)] =0
#     return accumulate_transition_probaN, cell_entry

# def pseudo_time(accumulate_transition_probaN, stationary_index, cell_entry):
#     max_element_index = np.argmax(cell_entry)
#     # root_cell = accumulate_transition_probaN[stationary_index]
#     root_cell = accumulate_transition_probaN[max_element_index]
#     pseudo_time = np.array([])
#     for i in range(len(accumulate_transition_probaN)):
#         pseudo_time = np.append(pseudo_time, np.linalg.norm(root_cell-accumulate_transition_probaN[i]))
#     pseudo_time = (pseudo_time - min(pseudo_time)) / (max(pseudo_time) - min(pseudo_time))
#     return pseudo_time

def amend_transition_proba_f(transition_proba):
    cell_entry = []
    accumulate_transition_probaN2 = transition_proba.copy()
    for item in accumulate_transition_probaN2:
        SCEntropy = sc_entry(item)
        cell_entry.append(SCEntropy)
    cell_entry = np.array(cell_entry)
    cell_entry_compar = np.zeros((len(cell_entry), len(cell_entry)))
    for i in range(len(cell_entry)):
        for j in range(len(cell_entry)):
            if cell_entry[i] > cell_entry[j]:
                # 如果细胞i的熵大于细胞j的熵，则可能发生转移
                cell_entry_compar[i,j] = 1
    accumulate_transition_probaN2[np.where(cell_entry_compar == 0)] =0
    return accumulate_transition_probaN2, cell_entry
def mfpt_f(transition_proba, adata):    
    amend_transition_proba, cell_entropy = amend_transition_proba_f(transition_proba)
    cell_entropy = (cell_entropy - min(cell_entropy)) / (max(cell_entropy) - min(cell_entropy))
    adata.obs['cell entropy'] = cell_entropy
    return amend_transition_proba
def PBA(adata, distances_transition_proba, transition_proba):
    cell_entropy = adata.obs['cell entropy'].values
    PBA_T = np.zeros((len(cell_entropy), len(cell_entropy)))
    i = 0
    for item1 in cell_entropy:
        j = 0
        for item2 in cell_entropy:
            if distances_transition_proba[i, j] != 0:
                if item1 > item2:
                    PBA_T[i, j] = np.exp(-(item1 - item2))*transition_proba[i, j]
                    # PBA_T[i, j] = np.exp(-(item1 - item2))
            j += 1
        # PBA_T[i, i] = 0
        i += 1
    PBA_T[PBA_T<0]=0
    for i in range(len(PBA_T)):
        if np.sum(PBA_T[i, :])!=0:
            PBA_T[i, :] = PBA_T[i, :] / np.sum(PBA_T[i, :])
    # sums = np.sum(PBA_T, axis=1)
    # PBA_T = np.where(sums[:, np.newaxis] != 0, PBA_T / sums[:, np.newaxis], 0)
    PBA_T[np.isnan(PBA_T)] = 0
    return PBA_T

# def Caculate_PseudoTime(PBA_T, root_index, RootGroup, adata):
#     '''
#     root_index: 表示根细胞的索引，格式为numpy.array
#     '''
#     accumulate_transition_probaN = np.linalg.inv(np.eye(len(PBA_T)) - PBA_T)
#     # FPT = np.dot(PBA_T, accumulate_transition_probaN)
#     matrix = accumulate_transition_probaN.copy()
#     RootNumber = len(root_index)
#     print(root_index)
#     print('RootNumber: ', RootNumber)
#     if RootNumber == 1:
#         pseudo_time = np.zeros(len(PBA_T))
#         # 仅有一个起点的数据集
#         root_cell = matrix[root_index[0]]
#         for i in range(len(matrix)):
#             pseudo_time[i] = np.linalg.norm(root_cell-matrix[i])
#         # pseudo_time[root_index[0]] = np.mean(pseudo_time)
#         pseudo_time = (pseudo_time - min(pseudo_time)) / (max(pseudo_time) - min(pseudo_time))
#         pseudo_time = pseudo_time.reshape(1, -1)
#     else:
#         # 包含多个起点的数据集
#         pseudo_time = np.zeros((RootNumber, len(PBA_T)))
#         RootGroupCell = []
#         for item in RootGroup:
#             RootGroupCell.append(list(np.where(adata.obs['GroupId'] == item)[0]))
#         i=0
#         for item1 in RootGroup:
#             # 针对每个起点进行单独计算
#             root_cell = matrix[root_index[i]]
#             for k in range(len(RootGroupCell)):
#                 if k!=i:
#                     pseudo_time[i, RootGroupCell[k]] = -1
#             j = 0
#             for item in matrix:
#                 if pseudo_time[i, j] != -1:
#                     pseudo_time[i, j] = np.linalg.norm(root_cell-item)
#                 j+= 1
#             pseudo_time_item = pseudo_time[i, :][pseudo_time[i, :] != -1]
#             pseudo_time[i, :][pseudo_time[i, :] != -1] = (pseudo_time_item - np.min(pseudo_time_item)) / (np.max(pseudo_time_item) - np.min(pseudo_time_item))
#             i+=1
#     return pseudo_time, accumulate_transition_probaN
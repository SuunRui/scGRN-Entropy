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


def read_GRN(svg_express_array, spot_num, hsvg_num, folder_way):
    '''通过读取文件获得GRNS相关信息'''
    GRNs_dim = np.empty([spot_num, hsvg_num])####存储GRN的度（基因对外调控，即基因的重要性）
    GRNs = np.empty([len(svg_express_array), len(svg_express_array[0]), len(svg_express_array[0])], dtype='float32')#初始化一个array来存储所有spot的GRN
    for i in range(len(svg_express_array)):
        file_way = folder_way + '/cell' + str(i+1) + '.txt'
        GRNs[i] = np.loadtxt(file_way)
        for j in range(len(GRNs[i])):
            GRNs[i][j][j] = 0
        #GRNs[np.where(np.abs(GRNs) < 0.01)] = 0######将GRNs的小于0.01的值设置为0
        # GRNs_dim[i] = np.sum(np.abs(GRNs[i]), axis = 0)
    return GRNs, GRNs_dim
def GRN_hotfig(GRN):
    # 获得GRN的in和out
    GRN_copy = np.copy(GRN)
    dim_in = np.sum(np.abs(GRN_copy), axis = 1)
    dim_out = np.sum(np.abs(GRN_copy), axis = 0)
    return dim_in, dim_out
def caculate_transition_proba(GRNs):
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


def save_distance(FolderName, result_path, data_path):
    pooled_data = pd.read_csv(result_path+FolderName+'/pooled_data.csv', index_col=0)
    # neighborsNumber = int(len(pooled_data) * 0.01)
    # if neighborsNumber < 10:
    #     print('数据集太小了')
    #     neighborsNumber = 10
    hvg_express_array = pooled_data.values
    GRNs, _ = read_GRN(hvg_express_array, len(hvg_express_array), len(hvg_express_array[0]), result_path + FolderName)
    transition_proba = caculate_transition_proba(GRNs)
    np.savetxt(result_path+FolderName+'/distance.txt', transition_proba)

# 启用R到pandas的数据框转换
# pandas2ri.activate()
# # 导入R的base包
# base = importr('base')
# # 指定文件夹路径
# data_path = 'data/silver/'
# result_path = 'result/silver_valve2/'
# # 获取文件夹中所有.rds文件
# rds_files = Path(data_path).rglob('*.rds')
# # 按顺序读取每个.rds文件
# for rds_file in sorted(rds_files):
#     # 读取.rds文件
#     rds_content = base.readRDS(str(rds_file))
#     # 获取.rds文件的名称（不包括扩展名）
#     rds_name = rds_file.stem
#     print(rds_name)
#     # 创建一个新的文件夹，名称为.rds文件的名称
#     new_folder_path = os.path.join(result_path, rds_name)
#     os.makedirs(new_folder_path, exist_ok=True)
#     preprocessing(rds_name,result_path, data_path)
    
    ##代码运行中，请勿关闭vscode
    ##代码运行中，请勿关闭vscode
    ##代码运行中，请勿关闭vscode
    ##代码运行中，请勿关闭vscode 
    ##代码运行中，请勿关闭vscode 
    ##代码运行中，请勿关闭vscode 
    ##代码运行中，请勿关闭vscode 
# preprocessing('planaria_parenchyme', 'result/', 'data_path')
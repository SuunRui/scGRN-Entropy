from gene_pool import True_pool
from GRN_func import GRN_func, read_GRN
from caculate_TranProb_PTime_Entropy import caculate_transition_proba, mfpt_f
from pseudo_time import PBA, preprocessing
from getMST import getmst, optimize_super_param
from distance_matrix import save_distance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.environ['R_HOME'] = 'D:/R-4.4.0'
import scanpy as sc
import anndata as ad
import matplotlib.colors as mcolors
import rpy2.robjects as robjects
from scipy.optimize import minimize
import networkx as nx
from pathlib import Path
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
import pcurve
import heapq
from scipy.spatial.distance import pdist, squareform


np.random.seed(1)
def main(FolderName, data_path, result_path):
    # Load the RDS file 
    rds_data = robjects.r['readRDS'](data_path + FolderName + '.rds')
    # 将R中的稀疏矩阵转换为Python中的稀疏矩阵
    dense_matrix = robjects.r['as'](rds_data.rx2('expression'), 'matrix')
    # 将稠密矩阵转换为NumPy数组
    dense_array = np.array(dense_matrix)
    # 获取稀疏矩阵的行名和列名
    rownames = list(rds_data.rx2('cell_info').rx2('cell_id'))
    colnames = list(rds_data.rx2('feature_info').rx2('feature_id'))
    adata = ad.AnnData(dense_array, obs = pd.DataFrame(index = rownames), var = pd.DataFrame(index = colnames))
    adata.obs['group_id'] = rds_data.rx2('grouping')
    adata.uns['milestone_network'] = rds_data.rx2('milestone_network')
    adata.uns['start_id'] = list(rds_data.rx2('prior_information').rx2('start_id'))
    adata.uns['start_milestones'] = list(rds_data.rx2('prior_information').rx2('start_milestones'))
    adata.obs['stage'] = np.array(rds_data.rx2('prior_information').rx2('timecourse_discrete'))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.filter_genes_dispersion(adata, n_top_genes = 2000)
    except:
        print('Have not enouth genes')

    neighborsNumber = int(len(adata.obs) * 0.1)

    if neighborsNumber < 10:
        print('数据集太小了')
        neighborsNumber = 10

    directory = result_path+FolderName
    if not os.path.exists(directory):
        os.makedirs(directory)
    # pooled_data = True_pool(adata,len(adata.var))
    # pooled_data.to_csv(result_path+FolderName+'/pooled_data.csv')
    pooled_data = pd.read_csv(result_path+FolderName+'/pooled_data.csv', index_col=0)
    hvg_express_array = pooled_data.values
    # _ = GRN_func(0, len(hvg_express_array[0]), hvg_express_array, result_path + FolderName)
    GRNs = read_GRN(hvg_express_array, len(hvg_express_array), len(hvg_express_array[0]), result_path + FolderName)
    transition_proba = caculate_transition_proba(GRNs)
    np.savetxt(result_path+FolderName+'/transition_proba.txt', transition_proba)
    transition_proba = np.loadtxt(result_path+FolderName+'/transition_proba.txt')
    _ = mfpt_f(transition_proba, adata)
    top_ten_indices = np.argsort(-transition_proba, axis=1)[:, :neighborsNumber]
    distances_transition_proba = np.zeros(transition_proba.shape)
    for i in range(len(top_ten_indices)):
        distances_transition_proba[i, top_ten_indices[i]] = 1
    PBA_T = PBA(adata, distances_transition_proba, transition_proba)
    np.savetxt(result_path+FolderName+'/PBA.txt', PBA_T)
    PBA_T = np.loadtxt(result_path+FolderName+'/PBA.txt')
    save_distance(FolderName, result_path, data_path)
    getmst(PBA_T, adata, FolderName, result_path, neighborsNumber, 0.3)


# data_path = "D:/A_study/A_study/cell_differentiation2/code/data/silvero/"
# FolderName = 'placenta-trophoblast-differentiation_mca'
# result_path = "D:/A_study/A_study/cell_differentiation2/code/result/silver_valve2/"
# main(FolderName, data_path, result_path)
# # 启用R到pandas的数据框转换

pandas2ri.activate()
# 导入R的base包
base = importr('base')
# 指定文件夹路径
data_path = "D:/A_study/A_study/cell_differentiation2/code/data/goldo/"
result_path = "D:/A_study/A_study/cell_differentiation2/code/result/gold_valve2/"
# 获取文件夹中所有.rds文件
rds_files = Path(data_path).rglob('*.rds')
# 按顺序读取每个.rds文件
for rds_file in sorted(rds_files):
    # 读取.rds文件
    rds_content = base.readRDS(str(rds_file))
    # 获取.rds文件的名称（不包括扩展名）
    rds_name = rds_file.stem
    print(rds_name)
    # 创建一个新的文件夹，名称为.rds文件的名称
    new_folder_path = os.path.join(result_path, rds_name)
    os.makedirs(new_folder_path, exist_ok=True)
    main(rds_name,data_path, result_path)
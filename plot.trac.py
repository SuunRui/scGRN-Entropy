import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import networkx as nx
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

def shortest_path_to_pseudo_time_group(G, start_group):
    G_copy = G.copy()
    new_mst_edges = []
    pseudo_time_df = pd.DataFrame(index=list(G_copy.nodes),columns=[str(i) for i in range(len(start_group))])
    i=0
    for item1 in start_group:
        for item2 in list(G_copy.nodes):
            if nx.has_path(G, item1, item2):
                shortest_path_item = nx.shortest_path(G, source=item1, target=item2, weight='weight')
                pseudo_time_df.loc[item2, str(i)] = len(shortest_path_item)
                if len(shortest_path_item) >1:
                    for i in range(len(shortest_path_item)-1):
                        new_mst_edges.append((shortest_path_item[i], shortest_path_item[i+1]))
            else:
                pseudo_time_df.loc[item2, str(i)] = 100
    row_min = pseudo_time_df.min(axis=1)
    row_min_values = row_min.values
    row_min_values = (row_min_values- np.min(row_min_values)) / (np.max(row_min_values)- np.min(row_min_values))
    pseudo_time_df['pseudo time'] = row_min_values
    row_min_df = pseudo_time_df.drop(columns=[str(i) for i in range(len(start_group))])
    new_mst_edges = list(set(new_mst_edges))
    return row_min_df, new_mst_edges



def plot_trajectory(adata, save_path, result_path, FolderName, neighborsNumber):
    # 设置字体属性
    font_properties = {
            'fontsize': 16,
            'fontweight': 'bold',
            'fontstyle': 'italic',
            'fontname': 'Times New Roman',
            'color': 'black'}
    
    #获取画图的数据
    sc.tl.pca(adata, svd_solver="arpack")
    # pca_data = adata.obsm['X_pca']
    sc.pp.neighbors(adata, n_neighbors=neighborsNumber)
    sc.tl.umap(adata)
    distance_all = np.loadtxt(result_path+FolderName+'/distance_all.txt')
    _, eigenvectors_distance_all = np.linalg.eig(distance_all)
    eigenvectors_distance_all = np.real(eigenvectors_distance_all)
    adata.obsm['X_eigenvectors'] = eigenvectors_distance_all[:, :2]
    adata.obsm['X_dif'] = adata.obsm['X_eigenvectors']/np.max(adata.obsm['X_eigenvectors']) + adata.obsm['X_umap'] / np.max(adata.obsm['X_umap'] )
    pseudo_time = np.loadtxt(result_path+FolderName+'/pseudo_time.txt')
    adata.obs['pseudo time'] = pseudo_time
    mst_Group_edges = np.loadtxt(result_path+FolderName+'\mst_Group.txt', dtype=str, comments=None, delimiter=',')
    # print(mst_Group_edges)
    mst_Group_edges_list = []
    print(mst_Group_edges.shape)
    for i in range(int(len(mst_Group_edges))):
        mst_Group_edges_list.append((mst_Group_edges[i, 0], mst_Group_edges[i, 1]))
    Group_mst = nx.Graph()
    Group_mst.add_edges_from(mst_Group_edges_list)
    start_Group = adata.uns['start_milestones']
    pseudo_time_Group, new_mst_edges = shortest_path_to_pseudo_time_group(Group_mst, start_Group)
    for item in adata.obs.index:
        adata.obs.loc[item, 'pseudo time'] = adata.obs.loc[item, 'pseudo time'] + pseudo_time_Group.loc[adata.obs.loc[item, 'group_id'], 'pseudo time']
    
    mean_position_umap = {}
    for item in adata.obs['group_id'].unique():
        indices = adata.obs.index[adata.obs['group_id'] == item].tolist()
        # adata_copy = adata[indices, ].obsm['X_dif']
        position = adata[indices, ].obsm['X_umap']
        mean_position_umap[item] = np.mean(position, axis=0)
    # mean_position_dif = {}
    # for item in adata.obs['group_id'].unique():
    #     indices = adata.obs.index[adata.obs['group_id'] == item].tolist()
    #     # adata_copy = adata[indices, ].obsm['X_dif']
    #     position = adata[indices, ].obsm['X_dif']
    #     mean_position_dif[item] = np.mean(position, axis=0)
    mean_position_dif = mean_position_umap
    # 创建图形和轴
    fig1, ax = plt.subplots()
    # 定义颜色和颜色映射
    colors = ['#1E469B','#2681B6', '#35B9C5','#96D2B0','#F9F8CA']
    cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
    
    # 绘制散点图
    scatter = ax.scatter(adata.obsm['X_umap'][:,0], adata.obsm['X_umap'][:,1], c=adata.obs['pseudo time'].values, cmap=cmap)
    
    # 绘制最小生成树的边
    for item in mst_Group_edges:
        start_x, start_y = mean_position_dif[item[0]][0], mean_position_dif[item[0]][1]
        end_x, end_y = mean_position_dif[item[1]][0], mean_position_dif[item[1]][1]
        
        # 在起点和终点添加文本标签
        # ax.text(start_x, start_y, item[0], fontdict=font_properties, verticalalignment='bottom', horizontalalignment='right')
        # ax.text(end_x, end_y, item[1], fontdict=font_properties, verticalalignment='bottom', horizontalalignment='right')
        
        # 绘制边
        plt.plot([start_x, end_x], [start_y, end_y], c='k')
    
    # 关闭坐标轴
    ax.axis('off')
    
    # 保存图像
    plt.savefig(os.path.join(save_path, 'Differentiation_trajectory_None.svg'), format='svg')
    # plt.show()

def main(FolderName, data_path, result_path, save_path):
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
    
    plot_trajectory(adata, save_path, result_path, FolderName, neighborsNumber)


data_path = "D:/A_study/A_study/cell_differentiation2/code/data/silver1/"
result_path = "D:/A_study/A_study/cell_differentiation2/code/result/silver_valve2/"
pandas2ri.activate()
# 导入R的base包
base = importr('base')
# 指定文件夹路径
# 获取文件夹中所有.rds文件
rds_files = Path(data_path).rglob('*.rds')
# 按顺序读取每个.rds文件
for rds_file in sorted(rds_files):
    save_path = "D:/A_study/A_study/cell_differentiation2/paper/paper/fig/result/"
    # 读取.rds文件
    rds_content = base.readRDS(str(rds_file))
    # 获取.rds文件的名称（不包括扩展名）
    rds_name = rds_file.stem
    print(rds_name)
    # 创建一个新的文件夹，名称为.rds文件的名称
    new_folder_path = os.path.join(result_path, rds_name)
    os.makedirs(new_folder_path, exist_ok=True)
    directory = save_path + str(rds_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_path = save_path + str(rds_name)
    main(rds_name,data_path, result_path, save_path)
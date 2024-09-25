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
import pcurve
import heapq
from scipy.spatial.distance import pdist, squareform


def normalize_f(x):
    y = np.empty(x.T.shape)
    i = 0
    for item in x.T:
        y[i] = (item-np.min(item))/(np.max(item)-np.min(item))
    return y.T
def tick(X):
    x = X*10
    if x != 0:
        y = x+1/x
    else:
        y=100
    return y
def PearsonCorr_func(x, y):
    # 计算均值
    X, Y = x.flatten(), y.flatten()
    mean_X = np.mean(X)
    mean_Y = np.mean(Y)
    # 计算协方差
    cov_XY = np.mean((X - mean_X) * (Y - mean_Y))
    # 计算标准差
    std_X = np.std(X, ddof=0)
    std_Y = np.std(Y, ddof=0)
    # 计算皮尔逊相关系数
    pearson_corr = cov_XY / (std_X * std_Y)
    return pearson_corr
def prim_with_start(G, start_node):
    mst = nx.Graph()
    visited = set([start_node])
    edges = [(data['weight'], start_node, to) for to, data in G[start_node].items()]
    heapq.heapify(edges)
    while edges:
        weight, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst.add_edge(frm, to, weight=weight)
            for next_to, data in G[to].items():
                if next_to not in visited:
                    heapq.heappush(edges, (data['weight'], to, next_to))
    return mst
def min_max(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))
import networkx as nx
import heapq
def adjacency_matrix_to_graph(matrix ,adata):
    G = nx.Graph()
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):  # 遍历上三角矩阵避免重复添加边
            if matrix[i][j] != 0:  # 忽略权重为0的边
                # if adata.obs['GroupId'][i] != adata.obs['GroupId'][j]:
                G.add_edge(adata.obs.index[i], adata.obs.index[j], weight=matrix[i][j])
    return G
def amend_matrix(distance_all, neighborsNumber):
    top_ten_indices = np.argsort(distance_all, axis=1)[:, :int(neighborsNumber)]
    distances_transition_proba = np.zeros(distance_all.shape)
    for i in range(len(top_ten_indices)):
        distances_transition_proba[i, top_ten_indices[i]] = 1
    distance_all_amend = np.zeros(distance_all.shape)
    i=0
    for item1 in distances_transition_proba:
        j=0
        for item2 in item1:
            if item2 != 0:
                distance_all_amend[i,j] = distance_all[i,j] 
            j+=1
        i+=1 
    return distance_all_amend, distances_transition_proba
def BuildNet(NodeName, adjacency_matrix, adata):
    # 创建一个无向图
    G = nx.Graph()
    edges = []
    i=0
    for item1 in NodeName:
        j=0
        for item2 in NodeName:
            if adjacency_matrix[i, j] != 0:
                edges.append((item1, item2, {'weight': adjacency_matrix[i, j]}))
            j+=1
        i+=1
    G.add_edges_from(edges)
    return G

def merge_nodes_with_weighted_edges(G, node1, node2):
    # 找到节点1和节点2分别连接到其他节点的边
    edges_to_merge = []
    dict1 = dict(G[node1].items())
    dict2 = dict(G[node2].items())
    nodesList = list(set(list(dict1.keys()) + list(dict2.keys())))
    if node1 in nodesList:
        nodesList.remove(node1)
    if node2 in nodesList:
        nodesList.remove(node2)
    for item in nodesList:
        if item in dict1.keys() and item in dict2.keys():
            # edges_to_merge.append((node1, item, {'weight':max(dict1[item]['weight'],dict2[item]['weight'])}))
            edges_to_merge.append((node1, item, {'weight':(dict1[item]['weight']+dict2[item]['weight'])/4}))
            # edges_to_merge.append((node1, item, {'weight':min(dict1[item]['weight'],dict2[item]['weight'])}))
        elif item in dict1.keys() and item not in dict2.keys():
            edges_to_merge.append((node1, item, {'weight':dict1[item]['weight']}))
        else:
            edges_to_merge.append((node1, item, {'weight':dict2[item]['weight']}))
    # 创建一个新的图，用于存储合并后的结果
    new_G = nx.Graph(G)
    # 删除原始的节点和边
    new_G.remove_node(node1)
    new_G.remove_node(node2)
    new_G.add_edges_from(edges_to_merge)
    return new_G
def CropNode(G, adata, RootGroup1, distance_all_amend_df):
    # 简化图
    group_id = adata.obs['group_id'].unique()
    group_number = len(group_id)
    # G = BuildNet([i for i in range(len(PBA_T))], PBA_T, adata)#构建无向图
    node_values = {} # 节点的伪时间的值
    node_length = {} # 合并节点的个数
    for item in group_id:
        cell_index_item = adata[adata.obs['group_id'] == item].obs.index
        Gitem = BuildNet(cell_index_item, distance_all_amend_df.loc[cell_index_item, cell_index_item].values, adata) # 针对Group构建子图
        connected_components = nx.connected_components(Gitem)  # Group子图的连通区域
        connected_components_list = [list(component) for component in connected_components]
        length = [len(lengthitem) for lengthitem in connected_components_list]
        if len(connected_components_list) != 0:
            for item1 in connected_components_list:
                if len(item1) == np.max(length): # 选取Group最大的子区域
                    # i=0
                    # for item2 in RootGroup1:
                    node_values[item] = np.mean(adata[item1, ].obs['pseudo time'].values)
                        # i+=1
                    node_length[item] = len(item1)
                    nodes_to_merge = item1.copy()
                    for j in nodes_to_merge[1:]:
                        # G = nx.contracted_nodes(G, item1[0], i)
                        G = merge_nodes_with_weighted_edges(G, item1[0], j)
                else:
                    for i in item1:
                        G.remove_node(i)
        else:
            item1 = cell_index_item.copy()
            node_values[item] = np.mean(adata[item1, ].obs['pseudo time'].values)
            node_length[item] = len(item1)
            nodes_to_merge = item1.copy()
            for j in nodes_to_merge[1:]:
                G = merge_nodes_with_weighted_edges(G, item1[0], j)
    new_names = {i: adata.obs.loc[i, 'group_id'] for i in G.nodes()}  # 例如，将节点 0 换成 Node_A，节点 1 换成 Node_B，以此类推
    # 重命名节点
    G = nx.relabel_nodes(G, new_names)
    # 删除图中自己与自己的边
    self_edges = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(self_edges)
    return G, node_values, node_length
def sc_entry(p):
    # 计算单细胞的熵
    entry_list = []
    for item in p:
        if item == 0:
            entry_list.append(0)
        else:
            entry_list.append(-item*np.log(item))
    return np.sum(entry_list)
def amend_transition_proba_f(accumulate_transition_probaN):
    cell_entry = []
    accumulate_transition_probaN2 = accumulate_transition_probaN.copy()
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
def Caculate_PseudoTime(PBA_T, root_index, RootGroup, adata):
    '''
    root_index: 表示根细胞的索引，格式为numpy.array
    '''
    accumulate_transition_probaN = np.linalg.inv(np.eye(len(PBA_T)) - PBA_T)
    # FPT = np.dot(PBA_T, accumulate_transition_probaN)
    matrix = accumulate_transition_probaN.copy()
    pseudo_time = np.zeros(len(PBA_T))
    # 仅有一个起点的数据集
    root_cell = matrix[root_index[0]]
    for i in range(len(matrix)):
        pseudo_time[i] = np.linalg.norm(root_cell-matrix[i])
    # pseudo_time[root_index[0]] = np.mean(pseudo_time)
    pseudo_time = (pseudo_time - np.min(pseudo_time) + 0.01 * (np.max(pseudo_time) - np.min(pseudo_time))) / (np.max(pseudo_time) - np.min(pseudo_time))
    # pseudo_time = pseudo_time.reshape(1, -1)
    return pseudo_time, accumulate_transition_probaN

# 将MST变成有向图
def directed_mst(mst, start):
    directed = nx.DiGraph()
    visited = set()

    def dfs(node):
        for neighbor in mst.neighbors(node):
            if neighbor not in visited:
                directed.add_edge(node, neighbor)
                visited.add(neighbor)
                dfs(neighbor)

    visited.add(start)
    dfs(start)
    return directed

def directed_mst_from_multiple_starts(mst, start_nodes):
    '''根据起点将最小生成树变成有向图'''
    directed = nx.DiGraph()
    visited = set()

    def dfs(node):
        for neighbor in mst.neighbors(node):
            if neighbor not in visited:
                directed.add_edge(node, neighbor)
                visited.add(neighbor)
                dfs(neighbor)

    for start in start_nodes:
        if start not in visited:
            visited.add(start)
            dfs(start)
    
    return directed

def According_StartNode_GetMST(G_Group, start_node):
    if len(start_node) == 1:
        mst_Group = nx.minimum_spanning_tree(G_Group, algorithm='kruskal', weight='weight')
        mst_Group = directed_mst(mst_Group, start_node)
    elif len(start_node) > 1:
        mst_Group_list = []
        for item in start_node:
            G_Group_copy = G_Group.copy()
            G_Group_copy.remove_node(item)
            mst_Group_item = nx.minimum_spanning_tree(G_Group_copy, algorithm='kruskal', weight='weight')
            mst_Group_list.append(mst_Group_item)
        mst_Group = nx.compose_all(mst_Group_list)
        # mst_Group = directed_mst_from_multiple_starts(mst_Group, start_node)
    else:
        print('Error, No Have Root Cell')
    return mst_Group
def getmst(PBA_T, adata, FolderName, result_path, neighborsNumber):

    transition_proba = np.loadtxt(result_path+FolderName+'/transition_proba.txt')
    distance_matrix = np.loadtxt(result_path+FolderName+'/distance.txt')

    GroupIdDict = {}
    i = 0
    for item in adata.obs['group_id'].unique():
        GroupIdDict[item] = i
        i+=1
    GroupId = []
    for item in adata.obs['group_id'].values:
        GroupId.append(int(GroupIdDict[item]))
    adata.obs['GroupId'] = GroupId

    sc.tl.pca(adata, svd_solver="arpack")
    pca_data = adata.obsm['X_pca']
    sc.pp.neighbors(adata, n_neighbors=neighborsNumber)
    sc.tl.umap(adata)
    # 计算样本之间的欧氏距离
    distances = pdist(pca_data, metric='euclidean')
    distances = squareform(distances)
    # np.savetxt(result_path+FolderName+'/distance_pca.txt', distances)
    # distances = np.loadtxt(result_path+FolderName+'/distance_pca.txt')
    distance_all = min_max(distances)*min_max(distance_matrix)
    # np.savetxt(result_path+FolderName+'/distance_all.txt', distance_all)
    # distance_all = np.loadtxt(result_path+FolderName+'/distance_all.txt')


    # adata.obsm['distance_all'] = distance_all
    RootGroup1 = []
    root_index = [np.where(adata.obs.index==item)[0][0] for item in adata.uns['start_id']]
    for item in adata.obs['GroupId'].values[root_index]:
        RootGroup1.append(item)
    distance_all_amend, distances_transition_proba = amend_matrix(distance_all, neighborsNumber)
    _ = mfpt_f(transition_proba, adata)
    PBA_T = PBA(adata, distances_transition_proba, transition_proba)
    # np.savetxt(result_path+FolderName+'/PBA.txt', PBA_T)
    # PBA_T = np.loadtxt(result_path+FolderName+'/PBA.txt')
    pseudo_time, _ = Caculate_PseudoTime(PBA_T, root_index, RootGroup1, adata)
    # np.savetxt(result_path+FolderName+'/pseudo_time.txt', pseudo_time)
    # pseudo_time = np.loadtxt(result_path+FolderName+'/pseudo_time.txt')
    adata.obs['pseudo time'] = pseudo_time
    distance_all_amend_df = pd.DataFrame(distance_all_amend, index=adata.obs.index, columns=adata.obs.index)
    G = adjacency_matrix_to_graph(distance_all_amend ,adata)
    # start_node = adata.uns['start_id'][0]
    # mst = prim_with_start(G, start_node)
    GroupIndex = []
    for item in adata.obs['GroupId'].unique():
        GroupIndex.append(np.where(adata.obs['GroupId'] == item)[0])
    G_Group, node_values, node_length = CropNode(G, adata[list(G.nodes),:], RootGroup1, distance_all_amend_df)
    node_values_list = list(node_values.values())
    max_node_values_list = np.max(node_values_list)
    min_node_values_list = np.min(node_values_list)
    for item in node_values.keys():
        node_values[item] = (node_values[item]-min_node_values_list)/(max_node_values_list-min_node_values_list)
    print(node_values)
    # pos_Group = nx.spring_layout(G_Group)
    # nx.draw_networkx_nodes(G_Group, pos_Group, node_size=300, alpha=1)
    # nx.draw_networkx_edges(G_Group, pos_Group, edge_color='k', width=1, alpha=0.3, arrowsize=3)
    # nx.draw_networkx_labels(G_Group, pos_Group, font_size=17, font_color='k', font_family='Times New Roman')
    # plt.title('G')
    # plt.axis('off')
    # plt.show()
    start_Group = adata.uns['start_milestones']
    max_weigth = 0
    min_weigth = 100000000
    for u,v,w in G_Group.edges(data=True):
        if w['weight']>max_weigth:
            max_weigth=w['weight']
        if w['weight']<min_weigth:
            min_weigth=w['weight']
    for u,v,w in G_Group.edges(data=True):
        w['weight'] = (w['weight']-min_weigth)/(max_weigth-min_weigth)
    difference_value = []
    for u,v in G_Group.edges:
        # difference_value.append(tick(np.abs(adata.obs.iloc[np.where(adata.obs['group_id'] == u)[0][0], 4] - adata.obs.iloc[np.where(adata.obs['group_id'] == v)[0][0], 4])))
        # w['weight'] = w['weight']
        difference_value.append(tick(np.abs(node_values[u] - node_values[v])))
    max_difference_value = np.max(difference_value)
    min_difference_value = np.min(difference_value)
    for u,v,w in G_Group.edges(data=True):
        # w['weight'] = w['weight']
        w['weight'] = w['weight'] + 0.2*(tick(np.abs(node_values[u] - node_values[v])) - min_difference_value)/(max_difference_value-min_difference_value)
    # mst_Group = According_StartNode_GetMST(G_Group, start_Group)
    try:
        mst_Group = According_StartNode_GetMST(G_Group, start_Group)
        print(mst_Group.edges())
        # mst_Group = nx.minimum_spanning_tree(G_Group, algorithm='kruskal', weight='weight')
        # mst_Group = prim_with_start(G_Group, start_Group)
        # mst_Group = directed_mst(mst_Group, start_Group)
        pos_Group = nx.spring_layout(mst_Group)
        nx.draw_networkx_nodes(mst_Group, pos_Group, node_size=300, alpha=1)
        nx.draw_networkx_edges(mst_Group, pos_Group, edge_color='k', width=1, alpha=0.3, arrowsize=3)
        nx.draw_networkx_labels(mst_Group, pos_Group, font_size=10, font_color='k', font_family='Times New Roman')
        plt.title('MST')
        plt.axis('off')
        plt.show()
        mst_Group_edges = list(mst_Group.edges)
        np.savetxt(result_path+FolderName+'\mst_Group.txt', mst_Group_edges, fmt='%s', delimiter=',')
        G_amend_edges = np.empty((0, 3))
        for u,v,w in list(G.edges(data=True)):
            if (adata.obs.loc[u, 'group_id'], adata.obs.loc[v, 'group_id']) not in mst_Group_edges and (adata.obs.loc[v, 'group_id'], adata.obs.loc[u, 'group_id']) not in mst_Group_edges:
                G.remove_edge(u,v)
            else:
                G_amend_edges=np.append(G_amend_edges, [u,v,w['weight']])
        np.savetxt(result_path+FolderName+'\G_edges.txt', G_amend_edges, fmt='%s', delimiter=',')
    except:
        print('disconnect')
    # for item_milestone_network in adata.uns['milestone_network']:
    #     print(item_milestone_network)
    edge_number = len(adata.uns['milestone_network'][0])
    print(adata.uns['milestone_network'])
    milestone_network = []
    for i in range(edge_number):
        milestone_network.append((adata.uns['milestone_network'][0][i], adata.uns['milestone_network'][1][i]))
    common_elements = set(mst_Group_edges) & set(milestone_network)
    count = len(common_elements)
    print('正确率：', count/edge_number)
    print(len(adata.uns['start_milestones']))
    # print(adata.uns['milestone_network'][0])#输出真实的细胞分化轨迹


# from rpy2.robjects import r, pandas2ri
# from rpy2.robjects.packages import importr
# from pathlib import Path
# np.random.seed(520)
# pandas2ri.activate()
# # 导入R的base包
# base = importr('base')
# # 指定文件夹路径
# data_path = 'data/silver1/'
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
#     getmst(data_path, rds_name, result_path)

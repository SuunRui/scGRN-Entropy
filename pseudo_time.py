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
                if item1 < item2:
                    PBA_T[i, j] = np.exp((item1 - item2))*transition_proba[i, j]
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
    RootNumber = len(root_index)
    print(root_index)
    print('RootNumber: ', RootNumber)
    if RootNumber == 1:
        pseudo_time = np.zeros(len(PBA_T))
        # 仅有一个起点的数据集
        root_cell = matrix[root_index[0]]
        for i in range(len(matrix)):
            pseudo_time[i] = np.linalg.norm(root_cell-matrix[i])
        # pseudo_time[root_index[0]] = np.mean(pseudo_time)
        pseudo_time = (pseudo_time - min(pseudo_time)) / (max(pseudo_time) - min(pseudo_time))
        pseudo_time = pseudo_time.reshape(1, -1)
    else:
        # 包含多个起点的数据集
        pseudo_time = np.zeros((RootNumber, len(PBA_T)))
        RootGroupCell = []
        for item in RootGroup:
            RootGroupCell.append(list(np.where(adata.obs['GroupId'] == item)[0]))
        i=0
        for item1 in RootGroup:
            # 针对每个起点进行单独计算
            root_cell = matrix[root_index[i]]
            for k in range(len(RootGroupCell)):
                if k!=i:
                    pseudo_time[i, RootGroupCell[k]] = -1
            j = 0
            for item in matrix:
                if pseudo_time[i, j] != -1:
                    pseudo_time[i, j] = np.linalg.norm(root_cell-item)
                j+= 1
            pseudo_time_item = pseudo_time[i, :][pseudo_time[i, :] != -1]
            pseudo_time[i, :][pseudo_time[i, :] != -1] = (pseudo_time_item - np.min(pseudo_time_item)) / (np.max(pseudo_time_item) - np.min(pseudo_time_item))
            i+=1
    return pseudo_time, accumulate_transition_probaN

def BuildNet(NodeName, adjacency_matrix, adata):
    # 创建一个无向图
    G = nx.Graph()
    # 添加节点
    NodeNameCopy = ['Group' + adata.obs['GroupId'].values[i] +'_'+ str(i) for i in NodeName]
    G.add_nodes_from(NodeNameCopy)
    # 添加无向边
    edges = []
    for i in NodeName:
        for j in NodeName:
            if adjacency_matrix[i, j] != 0:
                # edges.append((i, j, {'weight': adjacency_matrix[i, j]}))
                edges.append(('Group' + adata.obs['GroupId'].values[i] +'_'+ str(i), 
                              'Group' + adata.obs['GroupId'].values[j] +'_'+ str(j), 
                              {'weight': adjacency_matrix[i, j]})) # 权重表示链接两个点的概率
    G.add_edges_from(edges)
    return G

def BuildDirNet(NodeName, adjacency_matrix):
    # 创建一个you向图
    G = nx.DiGraph()
    # 添加节点
    G.add_nodes_from(NodeName)
    # 添加无向边
    edges = []
    i = 0
    for item1 in NodeName:
        j = 0
        for item2 in NodeName:
            if adjacency_matrix[item1, item2] != 0:
                edges.append((item1, item2, {'weight': adjacency_matrix[item1, item2]}))
            j += 1
        i += 1
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
            edges_to_merge.append((node1, item, {'weight':dict1[item]['weight'] + dict2[item]['weight']}))
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
def CropNode(PBA_T, GroupIndex, adata, RootGroup1):
    # 简化图
    G = BuildNet([i for i in range(len(PBA_T))], PBA_T, adata)#构建无向图
    node_values = [{} for _ in range(len(RootGroup1))] # 节点的伪时间的值
    # node_values = {} # 节点的伪时间的值
    node_length = {} # 合并节点的个数
    for item in GroupIndex:
        Gitem = BuildNet(item, PBA_T, adata) # 针对Group构建子图
        connected_components = nx.connected_components(Gitem)  # Group子图的连通区域
        connected_components_list = [list(component) for component in connected_components]
        length = [len(lengthitem) for lengthitem in connected_components_list]
        sumLength = np.sum(length)
        for item1 in connected_components_list:
            if len(item1) >= sumLength * 0.5: # 选取Group最大的子区域
                item1Index = [int(item2[item2.find('_')+1:]) for item2 in item1]
                i=0
                for item in RootGroup1:
                    node_values[i][item1[0]] = np.mean(adata.obs['pseudo time' + item][item1Index].values)
                    i+=1
                node_length[item1[0]] = len(item1)
                nodes_to_merge = item1
                for i in nodes_to_merge[1:]:
                    # G = nx.contracted_nodes(G, item1[0], i)
                    G = merge_nodes_with_weighted_edges(G, item1[0], i)
            else:
                for i in item1:
                    G.remove_node(i)
    # 删除图中自己与自己的边
    self_edges = [(u, v) for u, v in G.edges() if u == v]
    G.remove_edges_from(self_edges)
    return G, node_values, node_length

def CropDiGraph(G, node_colors, rootname, node_length, node_values):
    diGCopy = nx.DiGraph(G)# 无向图复制为有向图
    dict_edges = list(diGCopy.edges(data=True))
    MaxWeight = 0
    for u, v, w in dict_edges:
        if w['weight']/(node_length[u] + node_length[v]) > MaxWeight:
            MaxWeight = w['weight']/(node_length[u] + node_length[v])
    for u, v, w in dict_edges:
        if node_values[u] - node_values[v] < 0: # 若节点u的伪时间低于v
            # 则有向边保留。伪时间越接近，相连的概率越小，边的权重越大
            # w['weight'] = 1/(w['weight'] * (node_values[v] - node_values[u]))
            # w['weight'] = 1/(((w['weight']/(node_length[u]+node_length[v]))/MaxWeight)+((node_values[v] - node_values[u])))
            w['weight'] = 1/(((w['weight']/(node_length[u]+node_length[v]))/MaxWeight))
        else:
            diGCopy.remove_edge(u, v)
    ShortestPathItem = []
    for item0 in rootname:
        StartGroup = item0
        for item1, item2 in zip(diGCopy.nodes(), diGCopy.edges()(data=True)):
            # nx.shortest_path函数希望权重越小越好
            if item1 != StartGroup:
                try:
                    path = tuple(nx.shortest_path(diGCopy, source=StartGroup, target=item1, weight='weight'))
                    if len(path) > 2:
                        for i in range(len(path)):
                            if i+1 == len(path):
                                break
                            else:
                                ShortestPathItem.append((path[i], path[i+1], {'weight':diGCopy.get_edge_data(path[i], path[i+1])['weight']}))
                    else:
                        ShortestPathItem.append(tuple(nx.shortest_path(diGCopy, source=StartGroup, target=item1, weight='weight'))+(item2[2],))
                except:
                    continue
    diGCopy.remove_edges_from(list(diGCopy.edges))
    diGCopy.add_edges_from(ShortestPathItem)
    return diGCopy



def preprocessing(PBA_T, adata, result_path, FolderName):
    
    adata.obsm['PBA_T'] = PBA_T
    RootCellName = adata.uns['start_id']
    root_index = np.array([])
    for item in RootCellName:
        root_index = np.append(root_index, np.where(adata.obs.index == item)[0])
        root_index = root_index.astype(int)
    GroupIdDict = dict()
    i = 0
    for item in adata.obs['group_id'].unique():
        GroupIdDict[item] = i
        i += 1
    GroupId = []
    for item in adata.obs['group_id'].values:
        GroupId.append(str(GroupIdDict[item]))
    adata.obs['GroupId'] = GroupId
    RootGroup1 = []
    for item in adata.obs['GroupId'].values[root_index]:
        RootGroup1.append(item)
    pseudo_time, _ = Caculate_PseudoTime(PBA_T, root_index, RootGroup1, adata)
    i = 0
    for item in RootGroup1:
        adata.obs['pseudo time'+item] = pseudo_time[i]
        np.savetxt(result_path+FolderName + '/pseudo_time' + item + '.txt', adata.obs['pseudo time' + item])
        i+=1
    RootGroup2 = []
    for item in adata.obs['GroupId'].values[root_index]:
        RootGroup2.append('Group' + item + '_')
    # 每个Group的index
    GroupNumber = len(adata.obs['GroupId'].unique())
    GroupIndex = []
    for item in adata.obs['GroupId'].unique():
        GroupIndex.append(np.where(adata.obs['GroupId'] == item)[0])
    G, node_values, node_length = CropNode(PBA_T, GroupIndex, adata, RootGroup1)
    # G, node_values, node_length = CropNode(PBA_T, GroupIndex, adata, pseudo_time_name)
    nodes = list(G.nodes())
    item1Index = [int(item2[item2.find('_')+1:]) for item2 in nodes]
    # 提取节点值，并将其转换为列表
    values = adata.obs['GroupId'][item1Index].astype(int)
    # 设置颜色映射和离散化边界
    cmap = plt.cm.Spectral
    bounds = range(min(values), max(values)+len(nodes))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # 映射节点颜色
    node_colors = [cmap(norm(value)) for value in values]
    for item in nodes:
        for i in range(len(RootGroup2)):
            if RootGroup2[i] in item:
                RootGroup2[i] = item
        diGCopy = []
    i = 0
    our_edges_set = set()
    node_values_array = np.empty((len(RootGroup1), len(nodes)))
    node_length_array = np.empty((1, len(nodes)))
    for i in range(len(RootGroup1)):
        j =0
        for item in nodes:
            node_values_array[i, j] = node_values[i][nodes[j]]
            node_length_array[0, j] = node_length[nodes[j]]
            j += 1
    node_values_array = np.vstack((node_values_array, node_length_array))
    node_values_df = pd.DataFrame(node_values_array.T, index=nodes, columns=RootGroup2+['node_length'])
    node_values_df.to_csv(result_path+FolderName + '/node_values.csv', index=True)
    GroupIdDict_df = pd.DataFrame(index=list(GroupIdDict.keys()))
    i = 0
    for item in GroupIdDict_df.index:
        GroupIdDict_df.loc[item, 0] = GroupIdDict[item]
    GroupIdDict_df.columns=['GroupId']
    GroupIdDict_df.to_csv(result_path+FolderName + '/GroupIdDict.csv', index=True)
    i= 0
    G_edges = []
    for u,v,w in list(G.edges(data=True)):
        G_edges.append((u, v, w['weight']))
    np.savetxt(result_path+FolderName + '/G_edges.txt', G_edges, fmt='%s')
    for item in RootGroup2:
        diGCopyitem = CropDiGraph(G, node_colors, [RootGroup2[i]], node_length, node_values[i])
        diGCopy.append(diGCopyitem)
        # 将我们输出的结果转化为集合类型
        our_edges = list(diGCopyitem.edges).copy()
        our_edges_setitem = set()
        for item in our_edges:
            our_edges_setitem.add((item[0][:item[0].find('_')], item[1][:item[1].find('_')]))
        our_edges_set = our_edges_set | our_edges_setitem
        # 获取真实的标签
        GroupIdDictCopy = GroupIdDict.copy()
        for item in GroupIdDictCopy.keys():
            GroupIdDictCopy[item] = 'Group' + str(GroupIdDictCopy[item])
        milestone_network_edges = []
        milestone_network_nodes = []
        for item in pd.DataFrame(adata.uns['milestone_network']).T.values:
            try:
                milestone_network_edges.append((GroupIdDictCopy[item[0]], GroupIdDictCopy[item[1]]))
                if GroupIdDictCopy[item[0]] not in milestone_network_nodes:
                    milestone_network_nodes.append(GroupIdDictCopy[item[0]])
                if GroupIdDictCopy[item[1]] not in milestone_network_nodes:
                    milestone_network_nodes.append(GroupIdDictCopy[item[1]])
            except:
                continue
        milestone_network_edges_set = set(milestone_network_edges)
        # 真实标签的可视化
        milestone_network = nx.DiGraph()
        milestone_network.add_nodes_from(milestone_network_nodes)
        milestone_network.add_edges_from(milestone_network_edges)
        i += 1
    if milestone_network_edges_set == our_edges_set:
        np.savetxt(result_path + FolderName+'/True_or_False.txt', ["True"], fmt='%s')
    np.savetxt(result_path + FolderName+'/milestone_network_edges.txt', np.array(list(milestone_network_edges_set)), fmt='%s')
    if our_edges_set!={}:
        np.savetxt(result_path + FolderName+'/our_edges.txt', np.array(list(our_edges_set)), fmt='%s')
    np.savetxt(result_path + FolderName+'/trajectory_type.txt', list(rds_data.rx2('trajectory_type')), fmt='%s')
    #判断结果是否100%正确
    print('轨迹是否完全一致: ',milestone_network_edges_set == our_edges_set)
    print('our_edges_set:', our_edges_set)
    print('milestone_network_edges_set:', milestone_network_edges_set)

# 启用R到pandas的数据框转换
pandas2ri.activate()
# 导入R的base包
base = importr('base')
# 指定文件夹路径
data_path = 'data/goldo/'
result_path = 'result/gold_valve2/'
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
    preprocessing(rds_name, result_path, data_path)

    
    
    
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode
    #代码运行中，请勿关闭vscode 
    
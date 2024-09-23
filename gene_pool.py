# 对基因进行池化
############用pcs进行降维###########
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Times New Roman'
# plt.rcParams['font.size'] = 14
def gene_pool(adata, k):
    try:
        data = adata.X.toarray()
    except:
        data = adata.X
    m, n = data.shape
    correlation_matrix = np.corrcoef(data.T)
    # 选择每个组中最相关的k条数据，并将它们合并成m/k条新的数据
    grouped_data = np.empty((len(data), 0))
    group_indices = []
    for i in range(0, n):
        if i not in group_indices:
            group_indice = np.argsort(correlation_matrix[i, :])[-k:]  # 获取与第i条数据最相关的k条数据的索引
            for j in group_indice:
                group_indices.append(j)
            group_data = np.sum(data[:, group_indice], axis=1).reshape(len(data), 1)/len(group_indice)
            grouped_data = np.hstack((grouped_data, group_data))
    columns = ['Pool_Gene'+str(i+1) for i in range(len(grouped_data[0]))]
    pooled_data = pd.DataFrame(grouped_data, index = adata.obs.index, columns=columns)
    return pooled_data, len(grouped_data[0])
# 对基因进行最终的池化
def True_pool(adata, geneN):
    # 观察到pool后的基因的数量会收敛到某一个点
    # 通过设置不同的组别数来找到收敛的点
    pooled_geneN_list = []
    for k in range(10, int(geneN/2.5), 2):
        _, pooled_geneN = gene_pool(adata, k)
        pooled_geneN_list.append(pooled_geneN)
    plt.plot([k for k in range(10, int(geneN/2.5), 2)], pooled_geneN_list)
    # 计算移动平均值
    window_size = 5  # 移动平均窗口大小
    moving_avg = np.convolve(pooled_geneN_list, np.ones(window_size)/window_size, mode='valid')
    # 观察移动平均值的变化趋势
    convergence_threshold = 2 # 收敛阈值，根据需要调整
    for i in range(3, len(moving_avg)):
        convergenceNumber = i
        if abs(moving_avg[i] - moving_avg[i-3]) < convergence_threshold:
            convergence_value = moving_avg[i]
            break
    print('最相似的被池化的单组基因数目为：', convergenceNumber*2+10)
    print("池化后的基因数量:", convergence_value)
    pooled_data, _ = gene_pool(adata, convergenceNumber*2+10)
    return pooled_data
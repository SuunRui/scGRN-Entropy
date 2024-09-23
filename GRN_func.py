# 新------------------求解GRN
import numpy as np
import pandas as pd
import time
from scipy.optimize import minimize
from sklearn.decomposition import PCA

def sc_sg_GRN(sc_gene):#########单个细胞单个基因受到的调控关系函数
    # 定义二次规划目标函数
    def objective_function(b):
        #########x[:-1]表示x_i##########
        ########接受输入array格式
        return (sum(b[:] * sc_gene[:])) ** 2
    shape = [len(sc_gene)-1, len(sc_gene)]
    GRN = np.zeros(shape)
    circulation = [i for i in range(len(sc_gene)-1)] # 实际上的基因数目为len(sc_gene)-1
    for i in circulation:
        # 定义等式约束函数, b的长度为: len(sc_gene)
        def equality_constraint1(b):
            # A_(i,i+1) = -1
            return b[i+1] + 1
        def equality_constraint2(b):
            ########sum(A_ij)[j:1:50]=-1
            return sum(b[1:]) + 1
        # 初始猜测值
        initial_guess = np.zeros(len(sc_gene))
        # 定义等式约束
        equality_constraints = [
            {'type': 'eq', 'fun': equality_constraint1},
            {'type': 'eq', 'fun': equality_constraint2}
        ]
        # 求解二次规划问题
        result = minimize(objective_function, 
                          initial_guess, 
                          constraints=equality_constraints,
                          )
        if not result.success:
            print("Optimization did not converge. Maximum iterations reached.")
        GRN[i] = result.x
        # 输出结果
        #print("最优解：", result.x)
        # print("最优目标值：", result.fun)
        # print('进行了第{}次:'.format(i))
    return GRN[:, 1:]

def GRN_func(star, extend, svg_express_array, file_way):
    # 计算所有的spot的GRN
    x0 = np.ones(len(svg_express_array)).reshape(len(svg_express_array), 1)
    svg_express_array = np.hstack((x0, svg_express_array))
    gene_range = range(star, star + extend + 1)#选择一个构建基因调控网络的基因范围
    GRN_all = np.empty([len(svg_express_array), len(gene_range) - 1 , len(gene_range) - 1])#初始化一个array来存储所有spot的GRN
    for item in zip(svg_express_array, range(len(svg_express_array))):
        GRN_item = sc_sg_GRN(item[0])
        np.savetxt(file_way +'cell' + str(item[1]+1) + '.txt', GRN_item, fmt = '%s')
        #GRN_all[item[1]] = GRN_item
        print('完成了第{}个cell'.format(item[1]+1))
    return GRN_all


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
        GRNs_dim[i] = np.sum(np.abs(GRNs[i]), axis = 0)
    return GRNs, GRNs_dim

# def GRN_hotfig(GRN):
#     # 获得GRN的in和out
#     GRN_copy = np.copy(GRN)
#     dim_in = np.sum(np.abs(GRN_copy), axis = 1)
#     dim_out = np.sum(np.abs(GRN_copy), axis = 0)
#     return dim_in, dim_out

# def cos_sim(a, b):
#     # 余弦相似度
#     a_norm = np.linalg.norm(a)
#     b_norm = np.linalg.norm(b)
#     cos = np.dot(a,b)/(a_norm * b_norm)
#     return cos

# def GRNs_dim_sim_func(GRNs_dim, spot_num):
#     '''根据GRN的度来计算spot之间的相似性'''
#     simi_GRNs_dim = np.empty([spot_num, spot_num])
#     i = 0
#     for item1 in GRNs_dim:
#         j = 0
#         for item2 in GRNs_dim:
#             simi_GRNs_dim[i][j] = cos_sim(item1, item2)
#             j += 1
#         i += 1
#     return simi_GRNs_dim



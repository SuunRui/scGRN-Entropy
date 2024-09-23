from gene_pool import True_pool
from GRN_func import GRN_func
from caculate_TranProb_PTime_Entropy import caculate_transition_proba, mfpt_f
from pseudo_time import PBA, preprocessing
from getMST import getmst





# result_path = 
# FolderName = 
neighborsNumber = int(len(adata.obs) * 0.01)
if neighborsNumber < 10:
    print('数据集太小了')
    neighborsNumber = 10
pooled_data = True_pool(adata,len(adata.var))
# pooled_data.to_csv(result_path+FolderName+'/pooled_data.csv')
# pooled_data = pd.read_csv(result_path+FolderName+'/pooled_data.csv', index_col=0)
hvg_express_array = pooled_data.values
_ = GRN_func(0, len(hvg_express_array[0]), hvg_express_array, result_path + FolderName)
GRNs, GRNs_dim = read_GRN(hvg_express_array, len(hvg_express_array), len(hvg_express_array[0]), result_path + FolderName)
transition_proba = caculate_transition_proba(GRNs)
# np.savetxt(result_path+FolderName+'/transition_proba.txt', transition_proba)
# transition_proba = np.loadtxt(result_path+FolderName+'/transition_proba.txt')
_ = mfpt_f(transition_proba, adata)
top_ten_indices = np.argsort(-transition_proba, axis=1)[:, :neighborsNumber]
distances_transition_proba = np.zeros(transition_proba.shape)
for i in range(len(top_ten_indices)):
    distances_transition_proba[i, top_ten_indices[i]] = 1
PBA_T = PBA(adata, distances_transition_proba, transition_proba)
# np.savetxt(result_path+FolderName+'/PBA.txt', PBA_T)
getmst(PBA_T, adata, FolderName, result_path)
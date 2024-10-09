from caculate_TranProb_PTime_Entropy import caculate_transition_proba
from getMST import *
from distance_matrix import save_distance
from utils import *

import numpy as np


def main(data_name):
    os.environ['R_HOME'] = 'D:/R-4.4.0'
    dictory = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    result_path = os.path.join(dictory, "result")
    data_path = os.path.join(dictory, "data")
    data_path = os.path.join(data_path, data_name)
    save_path = os.path.join(result_path, data_name.split(".")[0])
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Read the Rds file and save it to adata
    adata, neighborsNumber = load_rds_to_anndata(data_path)

    # get super genes
    pooled_data = True_pool(adata, len(adata.var), save_path)
    hvg_express_array = pooled_data.values

    # caculate GRN
    # _ = GRN_func(0, len(hvg_express_array[0]), hvg_express_array, save_path)
    GRNs = read_GRN(hvg_express_array, save_path)

    # Computed transition probability
    transition_proba = caculate_transition_proba(GRNs)
    np.savetxt(save_path + "/transition_proba.txt", transition_proba)
    transition_proba = np.loadtxt(save_path+'/transition_proba.txt')
    _ = mfpt_f(transition_proba, adata)
    top_ten_indices = np.argsort(-transition_proba, axis=1)[:, :neighborsNumber]
    distances_transition_proba = np.zeros(transition_proba.shape)
    for i in range(len(top_ten_indices)):
        distances_transition_proba[i, top_ten_indices[i]] = 1

    # Calculate the corrected transition probability
    PBA_T = PBA(adata, distances_transition_proba, transition_proba)
    np.savetxt(save_path + "/PBA.txt", PBA_T)
    PBA_T = np.loadtxt(save_path+'/PBA.txt')
    save_distance(save_path)
    _ = getmst(PBA_T, adata, save_path, neighborsNumber, 0.3)

    # Visualize trajectories and pseudotimes
    plot_trajectory(adata, save_path, neighborsNumber)

if __name__ == '__main__':
    # get dataset name from floder
    data_name = "germline-human-female_li.rds"
    main(data_name)
import os
import numpy as np
import pandas as pd
import anndata as ad
from rpy2 import robjects
import scanpy as sc
from scipy.optimize import minimize
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import networkx as nx
os.environ['R_HOME'] = 'D:/R-4.4.0'
import rpy2.robjects as robjects


def load_rds_to_anndata(data_path):
    # Load the RDS file 
    rds_data = robjects.r['readRDS'](data_path)
    
    # Convert a sparse matrix in R to a sparse matrix in Python
    dense_matrix = robjects.r['as'](rds_data.rx2('expression'), 'matrix')
    
    # Converts a dense matrix to a NumPy array
    dense_array = np.array(dense_matrix)
    
    # Gets the row and column names of the sparse matrix
    rownames = list(rds_data.rx2('cell_info').rx2('cell_id'))
    colnames = list(rds_data.rx2('feature_info').rx2('feature_id'))
    
    # Create an AnnData object
    adata = ad.AnnData(dense_array, obs=pd.DataFrame(index=rownames), var=pd.DataFrame(index=colnames))
    
    # Add prior information
    adata.obs['group_id'] = rds_data.rx2('grouping')
    # adata.uns['milestone_network'] = rds_data.rx2('milestone_network')
    adata.uns['start_id'] = list(rds_data.rx2('prior_information').rx2('start_id'))
    adata.uns['start_milestones'] = list(rds_data.rx2('prior_information').rx2('start_milestones'))
    # adata.obs['stage'] = np.array(rds_data.rx2('prior_information').rx2('timecourse_discrete'))
    
    # preprocessing data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    try:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        sc.pp.filter_genes_dispersion(adata, n_top_genes=2000)
    except:
        print('Have not enough genes')
    
    neighborsNumber = int(len(adata.obs) * 0.1)
    if neighborsNumber < 10:
        neighborsNumber = 10
    
    return adata, neighborsNumber

# Pooling the genes
def gene_pool(adata, k):
    try:
        data = adata.X.toarray()
    except:
        data = adata.X
    m, n = data.shape
    correlation_matrix = np.corrcoef(data.T)
    # Select the most relevant k pieces of data from each group and merge them into m/k new pieces of data
    grouped_data = np.empty((len(data), 0))
    group_indices = []
    for i in range(0, n):
        if i not in group_indices:
            # Gets an index of the Article k data that is most relevant to Article i data
            group_indice = np.argsort(correlation_matrix[i, :])[-k:]  
            for j in group_indice:
                group_indices.append(j)
            group_data = np.sum(data[:, group_indice], axis=1).reshape(len(data), 1)/len(group_indice)
            grouped_data = np.hstack((grouped_data, group_data))
    columns = ['Pool_Gene'+str(i+1) for i in range(len(grouped_data[0]))]
    pooled_data = pd.DataFrame(grouped_data, index = adata.obs.index, columns=columns)
    return pooled_data, len(grouped_data[0])

# The ultimate pooling of genes
def True_pool(adata, geneN, save_path=None):
    pooled_geneN_list = []
    for k in range(10, int(geneN/2.5), 2):
        _, pooled_geneN = gene_pool(adata, k)
        pooled_geneN_list.append(pooled_geneN)
    plt.plot([k for k in range(10, int(geneN/2.5), 2)], pooled_geneN_list)
    # Calculating moving average
    window_size = 5  # Moving average window size
    moving_avg = np.convolve(pooled_geneN_list, np.ones(window_size)/window_size, mode='valid')
    # Observe the changing trend of the moving average
    convergence_threshold = 2 # The convergence threshold 
    for i in range(3, len(moving_avg)):
        convergenceNumber = i
        if abs(moving_avg[i] - moving_avg[i-3]) < convergence_threshold:
            convergence_value = moving_avg[i]
            break
    print('The most similar number of pooled genes is:', convergenceNumber*2+10)
    print("Numbers of supergene:", convergence_value)
    pooled_data, _ = gene_pool(adata, convergenceNumber*2+10)
    if save_path is not None:
        pooled_data.to_csv(save_path+'/pooled_data.csv')
    return pooled_data

def sc_sg_GRN(sc_gene):
    # The regulatory relationship function of single gene in single cell
    # Define the quadratic programming objective function
    def objective_function(b):
        # Accepts the input array format
        return (sum(b[:] * sc_gene[:])) ** 2
    shape = [len(sc_gene)-1, len(sc_gene)]
    GRN = np.zeros(shape)
    circulation = [i for i in range(len(sc_gene)-1)] # The actual number of genes is len(sc gene)-1
    for i in circulation:
        # Define the equality constraint function, b of length: len(sc gene)
        def equality_constraint1(b):
            return b[i+1] + 1
        def equality_constraint2(b):
            return sum(b[1:]) + 1
        initial_guess = np.zeros(len(sc_gene))
        # Defining equality constraint
        equality_constraints = [
            {'type': 'eq', 'fun': equality_constraint1},
            {'type': 'eq', 'fun': equality_constraint2}
        ]
        # Solve the quadratic programming problem
        result = minimize(objective_function, 
                          initial_guess, 
                          constraints=equality_constraints,
                          )
        if not result.success:
            print("Optimization did not converge. Maximum iterations reached.")
        GRN[i] = result.x
        
    return GRN[:, 1:]

# solve GRN
def GRN_func(star, extend, svg_express_array, file_way):
    # Calculate the GRN for all spots
    save_way = os.path.join(file_way, 'GRN')
    if not os.path.exists(save_way):
        os.makedirs(save_way)
    x0 = np.ones(len(svg_express_array)).reshape(len(svg_express_array), 1)
    svg_express_array = np.hstack((x0, svg_express_array))
    # Select a gene range to build a gene regulatory network
    gene_range = range(star, star + extend + 1)
    # Initializes an array to store the GRNS of all spots
    GRN_all = np.empty([len(svg_express_array), len(gene_range) - 1 , len(gene_range) - 1])
    for item in zip(svg_express_array, range(len(svg_express_array))):
        GRN_item = sc_sg_GRN(item[0])
        np.savetxt(save_way +'/cell' + str(item[1]+1) + '.txt', GRN_item, fmt = '%s')
        print('{} th GRN has completed'.format(item[1]+1))
    return GRN_all

# read GRN
def read_GRN(svg_express_array,  folder_way):
    folder_way = os.path.join(folder_way, 'GRN')
    GRNs = np.empty([len(svg_express_array), len(svg_express_array[0]), len(svg_express_array[0])], dtype='float32')
    for i in range(len(svg_express_array)):
        file_way = folder_way + '/cell' + str(i+1) + '.txt'
        GRNs[i] = np.loadtxt(file_way)
        for j in range(len(GRNs[i])):
            GRNs[i][j][j] = 0
    return GRNs

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



def plot_trajectory(adata, save_path, neighborsNumber):
    font_properties = {
            'fontsize': 9,
            'fontweight': 'bold',
            'fontstyle': 'italic',
            'fontname': 'Times New Roman',
            'color': 'black'}
    
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=neighborsNumber)
    sc.tl.umap(adata)
    distance_all = np.loadtxt(save_path+'/distance_all.txt')
    _, eigenvectors_distance_all = np.linalg.eig(distance_all)
    eigenvectors_distance_all = np.real(eigenvectors_distance_all)
    adata.obsm['X_eigenvectors'] = eigenvectors_distance_all[:, :2]
    adata.obsm['X_dif'] = adata.obsm['X_eigenvectors']/np.max(adata.obsm['X_eigenvectors']) + adata.obsm['X_umap'] / np.max(adata.obsm['X_umap'] )
    pseudo_time = np.loadtxt(save_path+'/pseudo_time.txt')
    adata.obs['pseudo time'] = pseudo_time
    mst_Group_edges = np.loadtxt(save_path+'/mst_Group.txt', dtype=str, comments=None, delimiter=',')
    mst_Group_edges_list = []
    for i in range(int(len(mst_Group_edges))):
        mst_Group_edges_list.append((mst_Group_edges[i, 0], mst_Group_edges[i, 1]))
    Group_mst = nx.Graph()
    Group_mst.add_edges_from(mst_Group_edges_list)
    start_Group = adata.uns['start_milestones']
    pseudo_time_Group, _ = shortest_path_to_pseudo_time_group(Group_mst, start_Group)
    for item in adata.obs.index:
        adata.obs.loc[item, 'pseudo time'] = adata.obs.loc[item, 'pseudo time'] + pseudo_time_Group.loc[adata.obs.loc[item, 'group_id'], 'pseudo time']
    mean_position_dif = {}
    for item in adata.obs['group_id'].unique():
        indices = adata.obs.index[adata.obs['group_id'] == item].tolist()
        position = adata[indices, ].obsm['X_dif']
        mean_position_dif[item] = np.mean(position, axis=0)
    fig1, ax = plt.subplots()
    colors = ['#1E469B','#2681B6', '#35B9C5','#96D2B0','#F9F8CA']
    cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
    scatter = ax.scatter(adata.obsm['X_dif'][:,0], adata.obsm['X_dif'][:,1], c=adata.obs['pseudo time'].values, cmap=cmap)
    cbar = fig1.colorbar(scatter, ax=ax)
    cbar.set_label('PseudoTime')
    cbar.set_ticks([])
    for item in mst_Group_edges:
        start_x, start_y = mean_position_dif[item[0]][0], mean_position_dif[item[0]][1]
        end_x, end_y = mean_position_dif[item[1]][0], mean_position_dif[item[1]][1]
        ax.text(start_x, start_y, item[0], fontdict=font_properties, verticalalignment='bottom', horizontalalignment='right')
        ax.text(end_x, end_y, item[1], fontdict=font_properties, verticalalignment='bottom', horizontalalignment='right')
        plt.plot([start_x, end_x], [start_y, end_y], c='k')
    ax.axis('off')
    plt.savefig(os.path.join(save_path, 'Differentiation_trajectory_None.svg'), format='svg')
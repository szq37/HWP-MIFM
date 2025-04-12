import os
from sklearn.model_selection import KFold
from param import *
from utils import *

args = parse_args()

def loading_similarity_feature(args):
    path = args.path
    device = args.device

    similarity_feature = {}

    "CircRNA Sequence Similarity"
    c_func_sim = np.loadtxt(os.path.join(path, 'circRNA_func.txt'), delimiter='\t', dtype=float)
    c_func_sim = torch.tensor(c_func_sim, device=device).to(torch.float32)
    c_func_edge_idx = get_edge_index(c_func_sim, device)
    c_func_graph = dgl.graph((c_func_edge_idx[0], c_func_edge_idx[1]))
    similarity_feature['c_func'] = {'Data': c_func_sim, 'Edge': c_func_edge_idx, 'Graph': c_func_graph}
    "CircRNA Gaussian Similarity"
    c_gauss_sim = np.loadtxt(os.path.join(path, 'circRNA_GIP.txt'), delimiter='\t', dtype=float)
    c_gauss_sim = torch.tensor(c_gauss_sim, device=device).to(torch.float32)
    c_gs_edge_idx = get_edge_index(c_gauss_sim, device)
    m_gs_graph = dgl.graph((c_gs_edge_idx[0], c_gs_edge_idx[1]))
    similarity_feature['c_gs'] = {'Data': c_gauss_sim, 'Edge': c_gs_edge_idx, 'Graph': m_gs_graph}

    "CircRNA Entropy Similarity"
    c_entro_sim = np.loadtxt(os.path.join(path, 'entropy_circRNA.txt'), delimiter='\t', dtype=float)
    c_entro_sim = torch.tensor(c_entro_sim, device=device).to(torch.float32)
    c_entro_edge_idx = get_edge_index(c_entro_sim, device)
    c_entro_graph = dgl.graph((c_entro_edge_idx[0], c_entro_edge_idx[1]))
    similarity_feature['c_entro'] = {'Data': c_entro_sim, 'Edge': c_entro_edge_idx, 'Graph': c_entro_graph}

    "CircRNA Cos Similarity"
    c_cos_sim = np.loadtxt(os.path.join(path, 'cos_circRNA.txt'), delimiter='\t', dtype=float)
    c_cos_sim = torch.tensor(c_cos_sim, device=device).to(torch.float32)
    c_cos_edge_idx = get_edge_index(c_cos_sim, device)
    c_cos_graph = dgl.graph((c_cos_edge_idx[0], c_cos_edge_idx[1]))
    similarity_feature['c_cos'] = {'Data': c_cos_sim, 'Edge': c_cos_edge_idx, 'Graph': c_cos_graph}

    "Disease Semantic Similarity"
    d_sem_sim = np.loadtxt(os.path.join(path, 'disease_semantic.txt'), delimiter='\t', dtype=float)
    d_sem_sim = torch.tensor(d_sem_sim, device=device).to(torch.float32)
    d_ss_edge_idx = get_edge_index(d_sem_sim, device)
    d_ss_graph = dgl.graph((d_ss_edge_idx[0], d_ss_edge_idx[1]))
    similarity_feature['d_ss'] = {'Data': d_sem_sim, 'Edge': d_ss_edge_idx, 'Graph': d_ss_graph}

    "Disease Gaussian Similarity"
    d_gauss_sim = np.loadtxt(os.path.join(path, 'disease_GIP.txt'), delimiter='\t', dtype=float)
    d_gauss_sim = torch.tensor(d_gauss_sim, device=device).to(torch.float32)
    d_gs_edge_idx = get_edge_index(d_gauss_sim, device)
    d_gs_graph = dgl.graph((d_gs_edge_idx[0], d_gs_edge_idx[1]))
    similarity_feature['d_gs'] = {'Data': d_gauss_sim, 'Edge': d_gs_edge_idx, 'Graph': d_gs_graph}

    "Disease Entropy Similarity"
    d_entro_sim = np.loadtxt(os.path.join(path, 'entropy_disease.txt'), delimiter='\t', dtype=float)
    d_entro_sim = torch.tensor( d_entro_sim, device=device).to(torch.float32)
    d_entro_edge_idx = get_edge_index( d_entro_sim, device)
    d_entro_graph = dgl.graph((d_entro_edge_idx[0], d_entro_edge_idx[1]))
    similarity_feature['d_entro'] = {'Data':  d_entro_sim, 'Edge': d_entro_edge_idx, 'Graph': d_entro_graph}

    "Disease Cos Similarity"
    d_cos_sim = np.loadtxt(os.path.join(path, 'cos_disease.txt'), delimiter='\t', dtype=float)
    d_cos_sim = torch.tensor( d_cos_sim, device=device).to(torch.float32)
    d_cos_edge_idx = get_edge_index( d_cos_sim, device)
    d_cos_graph = dgl.graph((d_cos_edge_idx[0], d_cos_edge_idx[1]))
    similarity_feature['d_cos'] = {'Data':  d_cos_sim, 'Edge': d_cos_edge_idx, 'Graph': d_cos_graph}

    average_miRNA_similarity =(c_cos_sim+c_entro_sim+c_func_sim+c_gauss_sim)/4
    similarity_feature['c_four_s'] = {'Data': average_miRNA_similarity}
    average_disease_similarity =(d_cos_sim+d_entro_sim+d_gauss_sim+d_sem_sim)/4
    similarity_feature['d_four_s'] = {'Data': average_disease_similarity}
    return similarity_feature
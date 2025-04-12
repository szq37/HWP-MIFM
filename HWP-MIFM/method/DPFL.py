import torch as th
import torch.nn as nn
from param import *
from dgl import function as fn
from utils import *
import dgl
import networkx as nx
from graph import *
from torch_geometric.nn import HypergraphConv
import numpy as np
import torch.nn.functional as F
args = parse_args()
class Linear(nn.Module):
    def __init__(self, args):
        super(Linear, self).__init__()
        n_rna = args.numrna
        n_dis = args.numdis
        self.mf = nn.Linear(n_rna, args.hidden)
        self.df = nn.Linear(n_dis, args.hidden)

    def forward(self, args, m_f, d_f):
        m_f = self.mf(m_f)
        d_f = self.df(d_f)
        return m_f, d_f
class Hyperrna(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Hyperrna, self).__init__()
        self.inputsize = input_size
        self.hiddensize = hidden_size
        self.outputsize = output_size
        self.conv1 = HypergraphConv(self.inputsize, self.hiddensize, use_attention=False, heads=8,
                                    concat=False, negative_slope=0.01, dropout=0.2, bias=True)
        self.conv2 = HypergraphConv(self.hiddensize, self.outputsize, use_attention=False, heads=8,
                                    concat=False, negative_slope=0.01, dropout=0.2, bias=True)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)

    def process_hypergraph(self, triplets, x):
        adj1_matrix = triplets
        adj1_matrix = adj1_matrix.clone().detach()
        hyperedge_index = torch.nonzero(adj1_matrix.T, as_tuple=True)
        hyperedge_index = torch.stack(hyperedge_index)
        hyperedge_index = hyperedge_index.to(x.device)
        output = self.conv1(x, hyperedge_index)
        output = F.relu(output)
        output = self.conv2(output, hyperedge_index)
        return output

    def forward(self, X, triplet):
        output = self.process_hypergraph(triplet, X)
        return output


class Hyperdis(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Hyperdis, self).__init__()
        self.inputsize = input_size
        self.hiddensize = hidden_size
        self.outputsize = output_size

        self.conv1 = HypergraphConv(self.inputsize, self.hiddensize, use_attention=False, heads=8,
                                    concat=False, negative_slope=0.01, dropout=0.2, bias=True)
        self.conv2 = HypergraphConv(self.hiddensize, self.outputsize, use_attention=False, heads=8,
                                    concat=False, negative_slope=0.01, dropout=0.2, bias=True)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif type(m) == nn.Conv2d:
                nn.init.uniform_(m.weight)

        self.conv1.apply(init_weights)
        self.conv2.apply(init_weights)

    def process_hypergraph(self, triplets, x):
        adj1_matrix = triplets
        adj1_matrix = adj1_matrix.clone().detach()
        hyperedge_index = torch.nonzero(adj1_matrix.T, as_tuple=True)
        hyperedge_index = torch.stack(hyperedge_index)
        hyperedge_index = hyperedge_index.to(x.device)
        output = self.conv1(x, hyperedge_index)
        output = F.relu(output)
        output = self.conv2(output, hyperedge_index)
        return output

    def forward(self, X, triplet):

        output = self.process_hypergraph(triplet, X)
        return output


class Trifusion(nn.Module):
    def __init__(self, args):
        super(Trifusion, self).__init__()
        self.n_rna = args.numrna
        self.n_dis = args.numdis
        self.Linear = Linear(args)
        self.hypermirna = Hyperrna(2 * args.numrna, args.hidden, args.hidden)
        self.hyperdis = Hyperdis(2 * args.numdis, args.hidden, args.hidden)
        self.linear_m = nn.Linear(2 * args.numrna, args.hidden)
        self.linear_d = nn.Linear(2 * args.numdis, args.hidden)
    def encode(self, args, similarity_feature):
        m_f = similarity_feature['c_four_s']['Data']
        d_f = similarity_feature['d_four_s']['Data']
        c_func = similarity_feature['c_func']['Data']
        c_gs = similarity_feature['c_gs']['Data']
        d_ss = similarity_feature['d_ss']['Data']
        dd_g = similarity_feature['d_gs']['Data']
        # second channel
        # rna section
        m_fea =torch.cat((c_func,c_gs), dim=1)
        m_hyper_fea1 = self.hypermirna(m_fea, m_f)
        m_hyper_fea2 = self.linear_m(m_fea)
        m_hyper_fea = (m_hyper_fea1 + m_hyper_fea2) / 2
        # dis section
        d_fea = torch.cat((d_ss, dd_g), dim=1)
        d_hyper_fea1 = self.hyperdis(d_fea, d_f)
        d_hyper_fea2 = self.linear_d(d_fea)
        d_hyper_fea = (d_hyper_fea1 + d_hyper_fea2) / 2
        hyper_fea = torch.cat((m_hyper_fea, d_hyper_fea), dim=0)
        return hyper_fea
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def save_features_to_file(features, filename,directory):
    filepath = os.path.join(directory, filename)
    np.savetxt(filepath, features.detach().cpu().numpy(), delimiter='\t', fmt='%.6f')
def main():
    args = parse_args()
    set_seed(args.SEED)
    similarity_feature = loading_similarity_feature(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Trifusion(args).to(args.device)
    hyper_fea = model.encode(args, similarity_feature)
    print(hyper_fea)
    hyper_circRNA_fea = hyper_fea[:249, :]
    hyper_disease_fea = hyper_fea[249:, :]
    directory = '../data/Dataset1'
    save_features_to_file(hyper_circRNA_fea, 'hyper_circRNA.txt',directory)
    save_features_to_file(hyper_disease_fea, 'hyper_disease.txt',directory)
    print(f"hyper_circRNA.txt saved with shape: {hyper_circRNA_fea.shape}")
    print(f"hyper_disease.txt saved with shape: {hyper_disease_fea.shape}")
if __name__ == "__main__":
    main()
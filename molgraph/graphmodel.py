######################
### Import Library ###
######################

# my library
from statistics import harmonic_mean
from molgraph.dataset import *
from molgraph.gineconv import *  # used modified version from torch_geometric
# standard
import numpy as np
import copy as copy
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, GRUCell
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import GATv2Conv  # <- 实际导入 GATv2Conv
from torch_geometric.nn import global_add_pool



# KAN
from molgraph.fourier_kan import KANLinear


######################
### Model Function ###
######################

# reset weight and load model
# Try resetting model weights to avoid weight leakage.
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def load_model(args):
    setting_param = dict()
    setting_param['file'] = args.file
    setting_param['model'] = args.model
    setting_param['schema'] = args.schema
    setting_param['reduced'] = args.reduced
    setting_param['batch_normalize'] = args.batch_normalize
    setting_param['device'] = args.device
    setting_param['dropout'] = args.dropout
    setting_param['use_kan_readout'] = args.use_kan_readout
    setting_param['kan_grid_size'] = args.kan_grid_size
    setting_param['kan_readout_norm'] = args.kan_readout_norm

    setting_param['use_kan_classifier'] = getattr(args, 'use_kan_classifier', True)


    setting_param['dict_reducedgraph'] = dict()
    if setting_param['schema'] in ['R', 'R_0', 'R_N', 'AR', 'AR_0', 'AR_N']:
        for g in setting_param['reduced']:
            setting_param['dict_reducedgraph'][g] = generateReducedGraphDict(args.file, g, vocab_file=args.file+'_0')

    NUM_FEATURES = {'atom': {'node': 79, 'edge': 10},
                    'junctiontree': {'node': 83, 'edge': 6},
                    'cluster': {'node': 83, 'edge': 6},
                    'functional': {'node': 115, 'edge': 20},
                    'pharmacophore': {'node': 6, 'edge': 3},
                    'substructure': {'node': 1024, 'edge': 14}}

    NUM_FEATURES_REDUCED = dict()
    for g in setting_param['reduced']:
        NUM_FEATURES_REDUCED[g] = dict()
        if setting_param['schema'] in ['R', 'AR']:
            NUM_FEATURES_REDUCED[g]['node'] = NUM_FEATURES[g]['node']
            NUM_FEATURES_REDUCED[g]['edge'] = NUM_FEATURES[g]['edge']
        elif setting_param['schema'] in ['R_0', 'AR_0']:
            NUM_FEATURES_REDUCED[g]['node'] = NUM_FEATURES['atom']['node']+NUM_FEATURES[g]['node']
            NUM_FEATURES_REDUCED[g]['edge'] = NUM_FEATURES[g]['edge']
        elif setting_param['schema'] in ['R_N']:
            # NUM_FEATURES_REDUCED[g]['node'] = args.out_channels 
            NUM_FEATURES_REDUCED[g]['node'] = NUM_FEATURES[g]['node']
            NUM_FEATURES_REDUCED[g]['edge'] = NUM_FEATURES[g]['edge']
        elif setting_param['schema'] in ['AR_N']:
            NUM_FEATURES_REDUCED[g]['node'] = args.out_channels+NUM_FEATURES[g]['node']
            # NUM_FEATURES_REDUCED[g]['node'] = NUM_FEATURES[g]['node']
            NUM_FEATURES_REDUCED[g]['edge'] = NUM_FEATURES[g]['edge']

    feature_graph_param = dict()
    feature_graph_param['node_graph_in_lin'] = NUM_FEATURES['atom']['node']
    feature_graph_param['node_graph_out_lin'] = args.in_channels
    feature_graph_param['edge_graph_in_lin'] = NUM_FEATURES['atom']['edge']
    feature_graph_param['edge_graph_out_lin'] = args.edge_dim

    feature_reduced_param = dict()
    for g in setting_param['reduced']:
        feature_reduced_param[g] = dict()
        feature_reduced_param[g]['node_reduced_in_lin'] = NUM_FEATURES_REDUCED[g]['node']
        feature_reduced_param[g]['node_reduced_out_lin'] = args.in_channels
        feature_reduced_param[g]['edge_reduced_in_lin'] = NUM_FEATURES_REDUCED[g]['edge']
        feature_reduced_param[g]['edge_reduced_out_lin'] = args.edge_dim

    graph_param = dict()
    graph_param['in_channels'] = args.in_channels
    graph_param['hidden_channels']= args.hidden_channels  
    graph_param['out_channels'] = args.out_channels 
    graph_param['num_layers'] = args.num_layers
    graph_param['in_lin'] = graph_param['out_channels']
    graph_param['out_lin'] = args.mol_embedding
    graph_param['num_layers_self'] = args.num_layers_self
    graph_param['edge_dim'] = args.edge_dim
    graph_param['heads'] = args.heads



    reduced_param = dict()
    for g in setting_param['reduced']:
        reduced_param[g] = dict()
        reduced_param[g]['in_channels'] = args.in_channels
        reduced_param[g]['hidden_channels'] = args.hidden_channels  
        reduced_param[g]['out_channels'] = args.out_channels
        reduced_param[g]['num_layers'] = args.num_layers_reduced # for reduced graph
        reduced_param[g]['in_lin'] = reduced_param[g]['out_channels'] 
        reduced_param[g]['out_lin'] = args.mol_embedding
        reduced_param[g]['num_layers_self'] = args.num_layers_self_reduced # for reduced graph
        reduced_param[g]['edge_dim'] = args.edge_dim
        reduced_param[g]['heads'] = args.heads

        
    
    linear_param = dict()
    if setting_param['schema'] in ['AR', 'AR_0', 'AR_N']:
        linear_param['out_lin'] = graph_param['out_lin']+sum([reduced_param[g]['out_lin'] for g in reduced_param])
    elif setting_param['schema'] in ['A']:
        linear_param['out_lin'] = graph_param['out_lin']
    elif setting_param['schema'] in ['R', 'R_0', 'R_N']:
        linear_param['out_lin'] = sum([reduced_param[g]['out_lin'] for g in reduced_param])
    
    linear_param['classification'] = args.class_number
    linear_param['batch_normalize'] = args.batch_normalize
    linear_param['dropout'] = args.dropout

   
    # print(setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param)
    model = GNN_Combine(setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param)
    
    # model.double()
    model.apply(reset_weights)

    return model


###################
### Model Class ###
###################

# Combine model
class GNN_Combine(torch.nn.Module):
    def __init__(self, setting_param, feature_graph_param, feature_reduced_param, graph_param, reduced_param, linear_param):
        super(GNN_Combine, self).__init__()

        self.file = setting_param['file']
        self.schema = setting_param['schema']
        self.reduced = setting_param['reduced']
        self.device = setting_param['device']
        self.dropout = setting_param['dropout']

        if self.schema in ['A', 'R_N', 'AR', 'AR_0', 'AR_N']:
            # features
            self.node_feature_graph = NodeLinear(setting_param,
                                                feature_graph_param['node_graph_in_lin'],
                                                feature_graph_param['node_graph_out_lin'])
            self.edge_feature_graph = EdgeLinear(setting_param,
                                                feature_graph_param['edge_graph_in_lin'],
                                                feature_graph_param['edge_graph_out_lin'])
            # graph
            self.GNN_Graph = GNN_Graph(setting_param,
                                    graph_param['in_channels'], 
                                    graph_param['hidden_channels'], 
                                    graph_param['out_channels'], 
                                    graph_param['num_layers'],
                                    graph_param['in_lin'], 
                                    graph_param['out_lin'], 
                                    graph_param['num_layers_self'],
                                    edge_dim=graph_param['edge_dim'],
                                    heads=graph_param['heads'])

        if self.schema in ['R', 'R_0', 'R_N', 'AR', 'AR_0', 'AR_N']:
            self.dict_reducedgraph = dict()
            self.node_feature_reduced = nn.ModuleDict()
            self.edge_feature_reduced = nn.ModuleDict()
            self.GNN_Reduced = nn.ModuleDict()
            
            for g in setting_param['reduced']:
                # dict of reduced
                self.dict_reducedgraph[g] = setting_param['dict_reducedgraph'][g]
                # feature
                self.node_feature_reduced[g] = NodeLinear(setting_param,
                                                        feature_reduced_param[g]['node_reduced_in_lin'],
                                                        feature_reduced_param[g]['node_reduced_out_lin'])
                self.edge_feature_reduced[g] = EdgeLinear(setting_param,
                                                        feature_reduced_param[g]['edge_reduced_in_lin'],
                                                        feature_reduced_param[g]['edge_reduced_out_lin'])
                # reduced
                self.GNN_Reduced[g] = GNN_Reduced(setting_param,
                                                  reduced_param[g]['in_channels'], 
                                                  reduced_param[g]['hidden_channels'], 
                                                  reduced_param[g]['out_channels'], 
                                                  reduced_param[g]['num_layers'],
                                                  reduced_param[g]['in_lin'], 
                                                  reduced_param[g]['out_lin'], 
                                                  reduced_param[g]['num_layers_self'],
                                                  edge_dim=reduced_param[g]['edge_dim'],
                                                  heads=reduced_param[g]['heads'])

        if self.schema in ['R_0', 'AR_0']:
            self.seen_collection = dict()
            for g in setting_param['reduced']:
                self.seen_collection[g] = dict()

        # pooling
        if self.schema in ['R_N']:
        # if self.schema in ['R_N', 'AR_N']:
            self.pool_features = nn.ModuleDict()
            self.pool_layer = dict()
            self.pool_conv = nn.ModuleDict()
            self.pool_bn = nn.ModuleDict()
            self.pool_gru = nn.ModuleDict()
            for g in setting_param['reduced']:
                self.pool_features[g] = Linear(feature_reduced_param[g]['node_reduced_out_lin'], graph_param['out_channels'])
                self.pool_layer[g] = graph_param['num_layers_self']
                self.pool_conv[g] = GATv2Conv(graph_param['out_channels'], graph_param['out_channels'], add_self_loops=False)

                self.pool_bn[g] = nn.BatchNorm1d(graph_param['out_channels'])
                self.pool_gru[g] = GRUCell(graph_param['out_channels'], graph_param['out_channels'])

        # classification
        # self.classification = ClassificationLayer(linear_param)
        self.classification = ClassificationLayer(linear_param)
        
        # last molecule embedding
        self.all_mol_embedding = dict()
        self.last_mol_embedding = None

        # attention molecule embedding
        self.all_mol_attention = dict()


    def generateReduced_pooling(self, g, data, graph, batch, fold_number=0):
        with torch.no_grad():
            sub_reduced = list()
            # batch_list = torch.bincount(batch).to(self.device) # non-deterministic
            batch_list = np.bincount(batch.cpu()) # deterministic
            graph_list = torch.split(graph, batch_list.tolist())
            for smiles, y, graphlist in zip(data.smiles, data.y, graph_list):
                # check smiles in seen collection?
                if self.schema in ['R_0', 'AR_0'] and smiles in self.seen_collection[g]:
                    _d = self.seen_collection[g][smiles]
                    d = copy.copy(_d)
                    sub_reduced.append(d)
                    continue
                # check smiles in dict_reducedgraph from dataset (train/val/test)
                if smiles in self.dict_reducedgraph[g]:
                    _d, cliques = self.dict_reducedgraph[g][smiles]
                # if not construct new graph
                else:
                    tokenizer = None
                    if g == 'substructure':
                        vocab_path = 'vocab/'+self.file+'_'+str(fold_number)+'.txt'
                        tokenizer = Tokenizer(vocab_path)
                    _d, cliques = constructReducedGraph(g, smiles, y, tokenizer)
                # check _d is valid
                if _d is not None:
                    d = copy.copy(_d)
                    x_r_new = torch.Tensor().to(self.device)
                    for s, c in zip(d.x_r, cliques):
                        indices = torch.tensor(c).to(self.device)
                        selected = torch.index_select(graphlist, 0, indices)
                        # max
                        # reducedmax = torch.max(selected, 0, True)[0]
                        # s_new = torch.cat((torch.unsqueeze(s.to(self.device), dim=0), reducedmax), -1)
                        # mean
                        # reducedmean = torch.mean(selected, 0, True)
                        # s_new = torch.cat((torch.unsqueeze(s.to(self.device), dim=0), reducedmean), -1)
                        # sum
                        reducedsum = torch.sum(selected, 0, True)
                        s_new = torch.cat((torch.unsqueeze(s.to(self.device), dim=0), reducedsum), -1)
                        x_r_new = torch.cat((x_r_new, s_new), 0)
                        # construct pooling index
                        if self.schema in ['AR_N']:
                            pooling_src = list()
                            pooling_dst = list()
                            for i, c in enumerate(cliques):
                                pooling_src.extend(c)
                                pooling_dst.extend([i]*len(c))
                            pooling_index = torch.tensor(np.array([pooling_src, pooling_dst])).to(self.device)
                            d.pooling_index = pooling_index 
                    d.x_r = x_r_new
                    sub_reduced.append(d)
                    # add processed smiles in seen_collection for R_0, AR_0 because of raw values
                    if self.schema in ['R_0', 'AR_0']: 
                        self.seen_collection[g][smiles] = d
                else:
                    print('ERROR: Generate Reduced Graph')
                    print(_d, cliques, smiles, y)
                    assert False
            # constructure dataloader from number of smiles
            loader = DataLoader(sub_reduced, batch_size=len(data.smiles), shuffle=False, follow_batch=['x_r'])
            data_r = next(iter(loader)).to(self.device)
            # 确保 reduced 图特征为 float32，避免与模型参数 dtype 冲突
            if hasattr(data_r, 'x_r') and data_r.x_r is not None:
                data_r.x_r = data_r.x_r.float()
            if hasattr(data_r, 'edge_attr_r') and data_r.edge_attr_r is not None:
                data_r.edge_attr_r = data_r.edge_attr_r.float()
            # if there is no edge in reduced graph
            if data_r.edge_attr_r.shape[0] == 0:
                data_r.edge_index_r = torch.tensor(np.array([[0],[0]])).type(torch.LongTensor).to(self.device)
            return data_r

    def generateReduced_only(self, g, data, fold_number=0):
        with torch.no_grad():
            sub = list()
            for smiles, y in zip(data.smiles, data.y):
                # check smiles in dict_reducedgraph from dataset (train/val/test)
                if smiles in self.dict_reducedgraph[g]:
                    _d, cliques = self.dict_reducedgraph[g][smiles]
                # if not construct new graph
                else:
                    tokenizer = None
                    if g == 'substructure':
                        vocab_path = 'vocab/'+self.file+'_'+str(fold_number)+'.txt'
                        tokenizer = Tokenizer(vocab_path)
                    _d, cliques = constructReducedGraph(g, smiles, y, tokenizer)
                # check _d is valid
                if _d is not None:
                    d = copy.copy(_d) 
                    # construct pooling index
                    if self.schema in ['R_N', 'AR_N']:
                        pooling_src = list()
                        pooling_dst = list()
                        for i, c in enumerate(cliques):
                            pooling_src.extend(c)
                            pooling_dst.extend([i]*len(c))
                        pooling_index = torch.tensor(np.array([pooling_src, pooling_dst])).to(self.device)
                        d.pooling_index = pooling_index
                    sub.append(d)
                else:
                    print('ERROR: Generate Reduced Graph')
                    print(_d, cliques, smiles, y)
                    assert False
            # constructure dataloader from number of smiles
            loader = DataLoader(sub, batch_size=len(data.smiles), shuffle=False, follow_batch=['x_r'])
            data_r = next(iter(loader)).to(self.device)
            # 确保 reduced 图特征为 float32，避免与模型参数 dtype 冲突
            if hasattr(data_r, 'x_r') and data_r.x_r is not None:
                data_r.x_r = data_r.x_r.float()
            if hasattr(data_r, 'edge_attr_r') and data_r.edge_attr_r is not None:
                data_r.edge_attr_r = data_r.edge_attr_r.float()
            # if there is no edge in reduced graph
            if data_r.edge_attr_r.shape[0] == 0:
                data_r.edge_index_r = torch.tensor(np.array([[0],[0]])).type(torch.LongTensor).to(self.device)
            return data_r

    def forward(self, data, return_attention_weights=False, fold_number=0):
        
        # 统一原子图特征/边特征为 float32，避免后续 Linear 里出现 Double/Float 混用
        if hasattr(data, "x_g") and data.x_g is not None:
            data.x_g = data.x_g.float()
        if hasattr(data, "edge_attr_g") and data.edge_attr_g is not None:
            data.edge_attr_g = data.edge_attr_g.float()

        # atom graph
        x_g_original = copy.copy(data.x_g)
        if self.schema in ['A', 'R_N', 'AR', 'AR_0', 'AR_N']:
            data.x_g = self.node_feature_graph(data.x_g)
            data.edge_attr_g = self.edge_feature_graph(data.edge_attr_g)
            graph_x, attention_weights_mol_g = self.GNN_Graph(data, data.x_g_batch)
            self.all_mol_embedding['atom'] = graph_x
            self.all_mol_attention['atom'] = attention_weights_mol_g

        # list of reduced graph
        for g in self.reduced:
            # reduced graph information/pooling
            if self.schema in ['R', 'AR']:
                data_r = self.generateReduced_only(g, data, fold_number=fold_number) # without atom layer info
            elif self.schema in ['R_N']:
            # elif self.schema in ['R_N', 'AR_N']:
                data_r = self.generateReduced_only(g, data, fold_number=fold_number) # without atom layer info for pooling
            elif self.schema in ['AR_N']:
                data_r = self.generateReduced_pooling(g, data, self.GNN_Graph.final_conv_acts, data.x_g_batch, fold_number=fold_number)
            elif self.schema in ['R_0', 'AR_0']:
                data_r = self.generateReduced_pooling(g, data, x_g_original, data.x_g_batch, fold_number=fold_number)

            # reduced graph
            if self.schema in ['R', 'R_0', 'AR', 'AR_0', 'AR_N']:
            # if self.schema in ['R', 'R_0', 'AR', 'AR_0']:
                data_r.x_r = self.node_feature_reduced[g](data_r.x_r)
                data_r.edge_attr_r = self.edge_feature_reduced[g](data_r.edge_attr_r)
                reduced_x, attention_weights_mol_r = self.GNN_Reduced[g](data_r, data_r.x_r_batch)
            elif self.schema in ['R_N']:
            # elif self.schema in ['R_N', 'AR_N']:
                data_r.x_r = self.node_feature_reduced[g](data_r.x_r)
                data_r.edge_attr_r = self.edge_feature_reduced[g](data_r.edge_attr_r)
                # pooling
                # out = F.relu(self.pool_features[g](data_r.x_r))
                out = self.pool_features[g](data_r.x_r)
                edge_index = data_r.pooling_index.to(torch.int64)
                for t in range(self.pool_layer[g]):
                    h = self.pool_conv[g]((self.GNN_Graph.final_conv_acts, out), edge_index)
                    if (h.size(0) != 1 and self.training) or (not self.training): h = self.pool_bn[g](h)
                    h = F.elu_(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    out = self.pool_gru[g](h, out)
                    out = F.leaky_relu(out)
                data_r.x_r = out
                reduced_x, attention_weights_mol_r = self.GNN_Reduced[g](data_r, data_r.x_r_batch)
            self.all_mol_embedding[g] = reduced_x
            self.all_mol_attention[g] = attention_weights_mol_r

        # before fully connected layer
        if self.schema in ['AR', 'AR_0', 'AR_N']:
            x = torch.cat([self.all_mol_embedding[emb] for emb in self.all_mol_embedding], dim=1)
        elif self.schema in ['A']:
            x = graph_x
        elif self.schema in ['R', 'R_0', 'R_N']:
            x = torch.cat([self.all_mol_embedding[emb] for emb in self.all_mol_embedding if emb != 'atom'], dim=1)

        # classification layer
        self.last_mol_embedding = x
        x = self.classification(x)
        
        # return
        if return_attention_weights:
            return x, self.all_mol_attention
        else: 
            return x
    
    def get_kan_layers(self):
        """
        收集模型中所有的 KAN 层
        返回: list of KANLinear layers
        """
        kan_layers = []
        
        # 从 GNN_Graph 收集
        if hasattr(self, 'GNN_Graph') and hasattr(self.GNN_Graph, 'lin'):
            if self.GNN_Graph.lin.__class__.__name__ == 'KANLinear':
                kan_layers.append(self.GNN_Graph.lin)
        
        # 从 GNN_Reduced 收集
        if hasattr(self, 'GNN_Reduced'):
            for g in self.GNN_Reduced:
                if hasattr(self.GNN_Reduced[g], 'lin'):
                    if self.GNN_Reduced[g].lin.__class__.__name__ == 'KANLinear':
                        kan_layers.append(self.GNN_Reduced[g].lin)
        
        # 从 classification 收集
        if hasattr(self, 'classification'):
            for name in ['lin1', 'lin2']:
                if hasattr(self.classification, name):
                    lin = getattr(self.classification, name)
                    if lin.__class__.__name__ == 'KANLinear':
                        kan_layers.append(lin)
        
        return kan_layers
    
    def get_molecule_embeddings(self):
        """
        获取最后的分子嵌入向量
        返回: torch.Tensor or None
        """
        return self.last_mol_embedding
    
    def get_attention_weights(self):
        """
        获取所有注意力权重
        返回: dict of attention weights
        """
        return self.all_mol_attention
    
    def extract_features(self, data):
        """
        提取特征用于可视化分析
        
        参数:
            data: 输入数据
        
        返回:
            features: numpy array of shape (batch_size, feature_dim)
        """
        self.eval()
        with torch.no_grad():
            _ = self.forward(data, return_attention_weights=False)
            embeddings = self.get_molecule_embeddings()
            if embeddings is not None:
                return embeddings.cpu().numpy()
        return None

# Node Embedding
class NodeLinear(nn.Module):
    def __init__(self, setting_param, in_lin, out_lin):
        super(NodeLinear, self).__init__()
        self.lin1 = Linear(in_lin, out_lin)
        self.batch_normalize = setting_param['batch_normalize']
        if self.batch_normalize: self.bn = nn.BatchNorm1d(out_lin)

    def forward(self, x):
           # x = x.double()
           x = self.lin1(x)
           if self.batch_normalize and ((x.size(0) != 1 and self.training) or (not self.training)): x = self.bn(x)
           x = F.leaky_relu(x)
           return x

# Edge Embedding
class EdgeLinear(nn.Module):
    def __init__(self, setting_param, in_lin, out_lin):
        super(EdgeLinear, self).__init__()
        self.lin1 = Linear(in_lin, out_lin)
        self.batch_normalize = setting_param['batch_normalize']
        if self.batch_normalize: self.bn = nn.BatchNorm1d(out_lin)

    def forward(self, x):
            if x.shape[0] != 0:
                # x = x.double()
                x = self.lin1(x)
                if self.batch_normalize and ((x.size(0) != 1 and self.training) or (not self.training)): x = self.bn(x)
                x = F.leaky_relu(x)
                return x
            else:
                return None

# Classification Layer
# class ClassificationLayer(nn.Module):
#     def __init__(self, linear_param):
#         super(ClassificationLayer, self).__init__()
#         out_lin = linear_param['out_lin']
#         classification = linear_param['classification']
#         # self.lin = Linear(out_lin, out_lin)
#         # self.bn = nn.BatchNorm1d(num_features=out_lin)
#         self.lin1 = Linear(out_lin, out_lin//2)
#         self.bn1 = nn.BatchNorm1d(num_features=out_lin//2)
#         self.lin2 = Linear(out_lin//2, out_lin//4)
#         self.bn2 = nn.BatchNorm1d(num_features=out_lin//4)
#         self.lin3 = Linear(out_lin//4, classification)
#         # self.lin3 = Linear(out_lin+(out_lin//4), classification)
#         self.dropout = linear_param['dropout']

#     def forward(self, x):
#         # wide
#         # h = self.lin(x)
#         h = x
#         # deep
#         # x = x.double()
#         x = self.lin1(x)
#         if (x.size(0) != 1 and self.training) or (not self.training):
#             x = self.bn1(x)
#         x = F.leaky_relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.lin2(x)
#         if (x.size(0) != 1 and self.training) or (not self.training):
#             x = self.bn2(x)
#         x = F.leaky_relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         out = self.lin3(x)
#         return out



# Classification Layer with Hybrid KAN
class ClassificationLayer(nn.Module):
    def __init__(self, linear_param, setting_param=None):
        super(ClassificationLayer, self).__init__()
        out_lin = linear_param['out_lin']
        classification = linear_param['classification']
        self.dropout = linear_param['dropout']
        
        # 从 setting_param 获取 KAN 配置
        use_kan = setting_param.get('use_kan_classifier', False) if setting_param else False
        grid_size = setting_param.get('kan_grid_size', 3) if setting_param else 3


        self.pre_norm = nn.BatchNorm1d(out_lin)
        
        if use_kan:
            # 第一层用 KAN 处理高维多视图特征
            self.lin1 = KANLinear(
                out_lin, 
                out_lin//2, 
                grid_size=grid_size,
                dropout=self.dropout,
                use_residual=False,
                adaptive_freq=True  # 自适应频率
            )
            self.bn1 = nn.BatchNorm1d(num_features=out_lin//2)
            
            # 后续层使用传统 Linear（更稳定）
            self.lin2 = Linear(out_lin//2, out_lin//4)
            self.bn2 = nn.BatchNorm1d(num_features=out_lin//4)
            self.lin3 = Linear(out_lin//4, classification)
            self.use_kan = 'hybrid'
        else:
            self.lin1 = Linear(out_lin, out_lin//2)
            self.bn1 = nn.BatchNorm1d(num_features=out_lin//2)
            self.lin2 = Linear(out_lin//2, out_lin//4)
            self.bn2 = nn.BatchNorm1d(num_features=out_lin//4)
            self.lin3 = Linear(out_lin//4, classification)
            self.use_kan = False

    def forward(self, x):

        # 1. 预处理
        if self.training and x.shape[0] == 1:
            # 训练时遇到单样本 batch，跳过 BN 运算以避免崩溃
            # 此时 x 保持原样传递，虽然分布可能不完美，但至少不会报错
            pass 
        else:
            x = self.pre_norm(x)
        # 第一层
        x = self.lin1(x)
        if (x.size(0) != 1 and self.training) or (not self.training):
            x = self.bn1(x)
        if self.use_kan == False:  # 仅在非 KAN 时添加激活
            x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层（混合和传统都一样）
        x = self.lin2(x)
        if (x.size(0) != 1 and self.training) or (not self.training):
            x = self.bn2(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 输出层
        out = self.lin3(x)
        return out


# class ClassificationLayer(nn.Module):
#     def __init__(self, linear_param, grid_size=3, kan_dropout=0.1):

#         super(ClassificationLayer, self).__init__()
#         out_lin = linear_param['out_lin']
#         classification = linear_param['classification']
#         mlp_dropout = linear_param['dropout']
        
#         # --- 1. 预归一化层 (Pre-Normalization) ---
#         # 强制将 GNN 的输出拉回到标准正态分布附近，这对于 Fourier 级数至关重要
#         self.pre_norm = nn.BatchNorm1d(out_lin)
        
#         # --- 2. 第一层: 混合过渡层 (Linear) ---
#         # 作用：稳定的降维，将特征压缩，避免直接用 KAN 处理高维输入带来的参数爆炸
#         self.lin1 = Linear(out_lin, out_lin // 2)
#         self.bn1 = nn.BatchNorm1d(out_lin // 2)
#         self.dropout_layer = nn.Dropout(p=mlp_dropout)
        
#         # --- 3. 第二层: Fourier-KAN (特征对齐) ---
#         # 作用：在低维空间进行复杂的非线性变换
#         self.kan2 = KANLinear(
#             out_lin // 2, 
#             out_lin // 4, 
#             grid_size=grid_size, 
#             dropout=kan_dropout, # KAN 内部的 dropout
#             use_residual=True,   # 开启残差连接防止梯度消失
#             adaptive_freq=True   # 允许自适应调整频率
#         )
        
#         # --- 4. 第三层: Fourier-KAN (最终决策) ---
#         self.kan3 = KANLinear(
#             out_lin // 4, 
#             classification, 
#             grid_size=grid_size, 
#             dropout=kan_dropout,
#             add_bias=True
#         )
        
#         # 手动初始化优化：确保 KAN 不会从剧烈的震荡开始
#         self._init_weights()

#     def _init_weights(self):
#         # 对 Linear 层使用 Kaiming 初始化
#         nn.init.kaiming_normal_(self.lin1.weight, mode='fan_in', nonlinearity='relu')
        
#         # KAN 层通常在内部已经初始化，但我们可以手动缩小最后一层的权重
#         # 使得初始预测接近 0，有助于分类任务的冷启动
#         with torch.no_grad():
#             self.kan3.weight.data *= 0.1 

#     def forward(self, x):
#         # 1. 预处理
#         if self.training and x.shape[0] == 1:
#             # 训练时遇到单样本 batch，跳过 BN 运算以避免崩溃
#             # 此时 x 保持原样传递，虽然分布可能不完美，但至少不会报错
#             pass 
#         else:
#             x = self.pre_norm(x)
        
#         # 2. Linear Stage (Wide -> Narrow)
#         x = self.lin1(x)
#         if (x.size(0) != 1 and self.training) or (not self.training):
#             x = self.bn1(x)
#         x = F.leaky_relu(x) # 这里的激活函数配合 Linear 使用
#         x = self.dropout_layer(x)
        
#         # 3. KAN Stage (Deep Nonlinear Interaction)
#         x = self.kan2(x)
        
#         # 4. Final Prediction
#         out = self.kan3(x)
        
#         return out

# Graph Layer
class GNN_Graph(nn.Module):
    def __init__(self, setting_param, in_channels, hidden_channels, out_channels, num_layers, in_lin, out_lin, num_layers_self, edge_dim, heads):
        super(GNN_Graph, self).__init__()
        self.device = setting_param['device']
        self.model = setting_param['model']
        self.dropout = setting_param['dropout']
        self.batch_normalize = setting_param['batch_normalize']
        self.num_layers = num_layers
        self.num_layers_self = num_layers_self

        if self.batch_normalize:
            self.convs, self.bns, self.atom_grus = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)
        else:
            self.convs, self.atom_grus = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)    
        

        # molecule
        self.mol_conv = GATv2Conv(out_channels, in_lin, add_self_loops=False).to(self.device)
        if self.batch_normalize: self.mol_bns = nn.BatchNorm1d(in_lin).to(self.device)
        self.mol_gru = GRUCell(in_lin, out_channels).to(self.device)

        # linear / Fourier-KAN readout
        self.lin, self.readout_norm = self._build_readout_layer(
            in_lin,
            out_lin,
            setting_param,
        )
        
        # explanation
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        
    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data, batch):

        x = data.x_g.float()
        edge_index = data.edge_index_g
        edge_attr = data.edge_attr_g.float() if hasattr(data, 'edge_attr_g') and data.edge_attr_g is not None else None
        edge_index = edge_index.type(torch.int64)

        self.input = x
        att_mol_stack = list()
        
        # Node Embedding:
        for _ in range(self.num_layers):
            h = self.convs[_](x, edge_index, edge_attr=edge_attr)
            if self.batch_normalize and ((h.size(0) != 1 and self.training) or (not self.training)): h = self.bns[_](h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.atom_grus[_](h, x)
            x = F.leaky_relu(x)
        self.final_conv_acts = x
        
        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = F.leaky_relu(global_add_pool(x, batch))

        for t in range(self.num_layers_self):
            h, attention_weights = self.mol_conv((x, out), edge_index, return_attention_weights=True)
            if self.batch_normalize and ((h.size(0) != 1 and self.training) or (not self.training)): h = self.mol_bns(h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out)
            out = F.leaky_relu(out)
            att_mol_index, att_mol_weights = attention_weights
            att_mol_stack.append(att_mol_weights)

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin(out)
        out = self._apply_readout_norm(out)
        # mean of attention weight
        att_mol_mean = torch.mean(torch.stack(att_mol_stack), dim=0)

        return out, (att_mol_index, att_mol_mean)
    
    def _build_readout_layer(self, in_dim, out_dim, setting_param):
        use_kan = bool(setting_param.get('use_kan_readout', False))
        grid_size = int(setting_param.get('kan_grid_size', 4))
        norm_type = setting_param.get('kan_readout_norm', 'none')
        self.use_kan_readout = use_kan
        self.kan_readout_norm = norm_type
        readout_norm = None

        if use_kan:
            layer = KANLinear(
                in_dim,
                out_dim,
                grid_size=grid_size,
                add_bias=True,
            )
            if norm_type == 'batchnorm':
                readout_norm = nn.BatchNorm1d(out_dim)
            elif norm_type == 'layernorm':
                readout_norm = nn.LayerNorm(out_dim)
        else:
            layer = Linear(in_dim, out_dim)

        return layer, readout_norm

    def _apply_readout_norm(self, out: torch.Tensor) -> torch.Tensor:
        if self.readout_norm is None:
            return out
        if isinstance(self.readout_norm, nn.BatchNorm1d):
            if (out.size(0) != 1 and self.training) or (not self.training):
                out = self.readout_norm(out)
            return out
        return self.readout_norm(out)



# Reduced Graph Layer
class GNN_Reduced(torch.nn.Module):
    def __init__(self, setting_param, in_channels, hidden_channels, out_channels, num_layers, in_lin, out_lin, num_layers_self, edge_dim, heads):
        super(GNN_Reduced, self).__init__()
        self.device = setting_param['device']
        self.model = setting_param['model']
        self.dropout = setting_param['dropout']
        self.batch_normalize = setting_param['batch_normalize']
        self.num_layers = num_layers
        self.num_layers_self = num_layers_self
        
        if self.batch_normalize:
            self.convs, self.bns, self.atom_grus = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)
        else:
            self.convs, self.atom_grus = GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads)    
        

        # molecule
        self.mol_conv = GATv2Conv(out_channels, in_lin, add_self_loops=False).to(self.device)
        if self.batch_normalize: self.mol_bns = nn.BatchNorm1d(in_lin).to(self.device)
        self.mol_gru = GRUCell(in_lin, out_channels).to(self.device)

        
        # linear / Fourier-KAN readout
        self.lin, self.readout_norm = self._build_readout_layer(
            in_lin,
            out_lin,
            setting_param,
        )
        
        # explanation
        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None
        
    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, data, batch):
        # 同样保证 reduced 图特征是 float32
        x = data.x_r.float()
        edge_index = data.edge_index_r
        edge_attr = data.edge_attr_r.float() if hasattr(data, 'edge_attr_r') and data.edge_attr_r is not None else None
        edge_index = edge_index.type(torch.int64)

        self.input = x
        att_mol_stack = list()
        
        # Node Embedding:
        for _ in range(self.num_layers):
            h = self.convs[_](x, edge_index, edge_attr=edge_attr)
            if self.batch_normalize and ((h.size(0) != 1 and self.training) or (not self.training)): h = self.bns[_](h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.atom_grus[_](h, x)
            x = F.leaky_relu(x)
        self.final_conv_acts = x
        
        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = F.leaky_relu(global_add_pool(x, batch))


        for t in range(self.num_layers_self):
            h, attention_weights = self.mol_conv((x, out), edge_index, return_attention_weights=True)
            if self.batch_normalize and ((h.size(0) != 1 and self.training) or (not self.training)): h = self.mol_bns(h)
            h = F.elu_(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out)
            out = F.leaky_relu(out)
            att_mol_index, att_mol_weights = attention_weights
            att_mol_stack.append(att_mol_weights)

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin(out)
        out = self._apply_readout_norm(out)
        # mean of attention weight
        att_mol_mean = torch.mean(torch.stack(att_mol_stack), dim=0)
        
        return out, (att_mol_index, att_mol_mean)
    
    def _build_readout_layer(self, in_dim, out_dim, setting_param):
        use_kan = bool(setting_param.get('use_kan_readout', False))
        grid_size = int(setting_param.get('kan_grid_size', 4))
        norm_type = setting_param.get('kan_readout_norm', 'none')
        self.use_kan_readout = use_kan
        self.kan_readout_norm = norm_type
        readout_norm = None

        if use_kan:
            layer = KANLinear(
                in_dim,
                out_dim,
                grid_size=grid_size,
                add_bias=True,
            )
            if norm_type == 'batchnorm':
                readout_norm = nn.BatchNorm1d(out_dim)
            elif norm_type == 'layernorm':
                readout_norm = nn.LayerNorm(out_dim)
        else:
            layer = Linear(in_dim, out_dim)

        return layer, readout_norm

    def _apply_readout_norm(self, out: torch.Tensor) -> torch.Tensor:
        if self.readout_norm is None:
            return out
        if isinstance(self.readout_norm, nn.BatchNorm1d):
            if (out.size(0) != 1 and self.training) or (not self.training):
                out = self.readout_norm(out)
            return out
        return self.readout_norm(out)


def GNN_Conv(setting_param, in_channels, hidden_channels, out_channels, num_layers, edge_dim, heads):
    device = setting_param['device']
    model = setting_param['model']
    batch_normalize = setting_param['batch_normalize']
    dropout = setting_param['dropout']
    num_layers = num_layers
    heads = heads

    convs = nn.ModuleList()
    if batch_normalize: bns = nn.ModuleList()
    atom_grus = nn.ModuleList()

    if model == 'GAT':
    
        # first layer
        convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, add_self_loops=True).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(heads * hidden_channels).to(device))
        atom_grus.append(GRUCell(heads * in_channels, hidden_channels).to(device))

        # hidden layer
        for _ in range(num_layers - 2):
            convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=edge_dim, add_self_loops=True).to(device))
            if batch_normalize: bns.append(nn.BatchNorm1d(heads * hidden_channels).to(device))
            atom_grus.append(GRUCell(heads * hidden_channels, hidden_channels).to(device))
            
        # last layer
        convs.append(GATv2Conv(hidden_channels, out_channels, heads=heads, concat=False, edge_dim=edge_dim, add_self_loops=True).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(out_channels).to(device))
        atom_grus.append(GRUCell(hidden_channels, out_channels).to(device))

    
    elif model == 'GIN':


        # first layer
        lin_gin = nn.Sequential(GIN_Sequential(in_channels, hidden_channels, dropout))
        convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
        atom_grus.append(GRUCell(in_channels, hidden_channels).to(device))

        # hidden layer
        for _ in range(num_layers - 2):
            lin_gin = nn.Sequential(GIN_Sequential(hidden_channels, hidden_channels, dropout))
            convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
            if batch_normalize: bns.append(nn.BatchNorm1d(hidden_channels).to(device))
            atom_grus.append(GRUCell(hidden_channels, hidden_channels).to(device))
        
        # last layer
        lin_gin = nn.Sequential(GIN_Sequential(hidden_channels, out_channels, dropout))
        convs.append(GINEConv(lin_gin, edge_dim=edge_dim).to(device))
        if batch_normalize: bns.append(nn.BatchNorm1d(out_channels).to(device))
        atom_grus.append(GRUCell(hidden_channels, out_channels).to(device))


    if batch_normalize:
        return convs, bns, atom_grus
    else:
        return convs, atom_grus


class GIN_Sequential(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout):
        super(GIN_Sequential, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.lin1 = Linear(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.leakyrelu1 = nn.LeakyReLU()
        self.lin2 = Linear(hidden_channels, hidden_channels)
        # self.leakyrelu2 = nn.LeakyReLU()
        self.dropout = dropout

    def forward(self, x):
        x = self.lin1(x)
        if (x.size(0) != 1 and self.training) or (not self.training): x = self.bn1(x)
        x = self.leakyrelu1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x
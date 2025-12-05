# My library
from molgraph.dataset import *
from molgraph.graphmodel import *
from molgraph.training import *
from molgraph.testing import *
from molgraph.visualize import *
from molgraph.experiment import *
from molgraph.interpret import *
from molgraph.fragmentation import *
# General library
import argparse
import numpy as np
import warnings
# pytorch
import torch
import pytorch_lightning as pl

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False


class MolPrediction:
    def __init__(self, fold, smiles, split, prediction, truevalue, attention):
        self.fold = fold
        self.smiles_original = smiles
        self.smiles = mol_to_smiles(smiles_to_mol(smiles))
        self.mol = smiles_to_mol(self.smiles)
        self.split = split
        self.prediction = prediction
        self.truevalue = truevalue
        self.attention = attention

    def print_prediction(self):
        print("-Molecule Prediction-"+"\n"+ \
        "Fold: "+str(self.fold)+"\n"+ \
        "SMILES_original: "+str(self.smiles_original)+"\n"+ \
        "SMILES: "+str(self.smiles)+"\n"+ \
        "Split: "+str(self.split)+"\n"+ \
        "Prediction: "+str(self.prediction)+"\n"+ \
        "TrueValue: "+str(self.truevalue))

    def print_attention(self):
        print("-Molecule Attention-")
        print_attention = list()
        for a in self.attention:
            print_attention.append(print(a))

    def __str__(self):
        self.print_prediction()
        self.print_attention()
        return "=====Complete!====="

    def to_dataframe(self):
        d = {'Fold': [self.fold for i in range(len(self.attention))],
        'SMILES_original': [self.smiles_original for i in range(len(self.attention))],
        'SMILES': [self.smiles for i in range(len(self.attention))],
        'Split': [self.split for i in range(len(self.attention))],
        'Prediction': [self.prediction for i in range(len(self.attention))],
        'TrueValue': [self.truevalue for i in range(len(self.attention))],
        'Schema': [a.schema for a in self.attention], 
        'Node_ID': [a.node_id for a in self.attention], 
        'Node_Feature': [a.node_feature for a in self.attention], 
        'Weight': [a.weight for a in self.attention], 
        'Weight_Other': [a.weight_other for a in self.attention]}
        df = pd.DataFrame(data=d)
        return df

class MolAttention:
    def __init__(self, schema, node_id, node_feature, weight=np.NaN, weight_other=np.NaN):
        self.schema = schema
        self.node_id = node_id
        self.node_feature = node_feature
        self.weight = weight
        self.weight_other = weight_other

    def __str__(self):
        print_attention = "Schema: "+str(self.schema)+"\t"+ \
        "Node_ID: "+str(self.node_id)+"\t"+ \
        "Node_Feature: "+str(self.node_feature)+"\t"+ \
        "Weight: "+str(self.weight)
        return print_attention
    
class MolAttentionSubstructure:
    def __init__(self, schema, node_id, node_feature, weight_sub=np.NaN, weight_other=np.NaN):
        self.schema = schema
        self.node_id = node_id
        self.node_feature = node_feature
        self.weight = weight_sub
        self.weight_other = weight_other

    def __str__(self):
        print_attention = "Schema: "+str(self.schema)+"\t"+ \
        "Node_ID: "+str(self.node_id)+"\t"+ \
        "Node_Feature: "+str(self.node_feature)+"\t"+ \
        "Weight_Sub: "+str(self.weight)+"\t"+ \
        "Weight_Other: "+str(self.weight_other)
        return print_attention


def getPredictionFold(args, args_test, all_dataset, datasets_splitted=None, print_result=False):
    
    molprediction = []

    # loop all fold
    for fold_number in range(args.fold):
    # for fold_number in tqdm([4]):
        args_test['fold_number'] = fold_number
        tester = Tester(args, args_test, print_model=False)

        for d in tqdm(all_dataset, desc='getMaskGraph'):
            # data loader
            test_loader, datasets_test =  generateDataLoaderListing([all_dataset[d]], 1)
            molecule_test = datasets_test[0]

            # molecule
            sample_graph = molecule_test
            smiles = sample_graph.smiles
            # mol = Chem.MolFromSmiles(smiles)

            molattention = []
            
            # testing
            try:
                spilttingdataset = 'test' if molecule_test.smiles in list(datasets_splitted[1]['X']) else 'train'
            except:
                spilttingdataset = '-'

            # true value
            truevalue = molecule_test.y
            try:
                truevalue = molecule_test.y.item()
            except:
                truevalue = molecule_test.y[0][0]

            # testing
            # print(datasets_test)
            try:
                predicted = tester.test_single(test_loader, return_attention_weights=True, print_result=False, raw_prediction=True)
                try:
                    predicted = predicted.item()
                except:
                    predicted = predicted[0][0]
            except Exception as e:
                print(e)
                predicted = None

            # print(predicted)
            # if predicted != molecule_test.y:
            #     print(molecule_test.smiles, "TRUE:", molecule_test.y, "PREDICTED:", predicted)

            if predicted is not None:

                try:
                    # attention result
                    att_mol = tester.getAttentionMol()
                    sample_att = att_mol
                    if 'atom' in sample_att:
                        sample_att_g = sample_att['atom']
                    else:
                        sample_att_g = None
                    if len(args.reduced) >= 1:
                        # Use the first reduced graph for mask_graph_r computation
                        reduced_key = args.reduced[0]
                        if reduced_key in sample_att:
                            sample_att_r = sample_att[reduced_key]
                        else:
                            # Try to find any available reduced graph key
                            available_keys = [k for k in sample_att.keys() if k != 'atom']
                            if available_keys:
                                sample_att_r = sample_att[available_keys[0]]
                                warnings.warn(f"Reduced graph '{reduced_key}' not found in attention. Using '{available_keys[0]}' instead.")
                            else:
                                sample_att_r = None
                                warnings.warn(f"No reduced graph attention found. Skipping reduced graph processing.")
                    else:
                        sample_att_r = None
                    # sample_att_g, sample_att_r = sample_att
                    if args.schema in ['A', 'R_N', 'AR', 'AR_0', 'AR_N']:
                        mask_graph_g = mask_graph(sample_att_g)
                    if args.schema in ['R', 'R_0', 'R_N', 'AR', 'AR_0', 'AR_N']:
                        mask_graph_r = mask_reduced(sample_att_r)
                    
                    mask_graph_x = None

                    # atom graph
                    if 'A' in args.schema:
                        reduced_graph, cliques, edges = getReducedGraph(args, ['atom'], smiles, normalize=False)
                        for i, f in enumerate(reduced_graph.node_features):
                            f_tuple = getImportanceFeatures(['atom'], f)
                            # Use .get() for safe dictionary access with warning for missing keys
                            if i not in mask_graph_g['atom']:
                                warnings.warn(f"Atom index {i} not found in mask_graph_g. Using default weight 0.0.")
                            weight = mask_graph_g['atom'].get(i, 0.0)
                            molattention.append(MolAttention('atom', i, f_tuple, weight))

                    # reduced graph
                    if 'R' in args.schema:
                        for r in args.reduced:
                            reduced_graph, cliques, edges = getReducedGraph(args, [r], smiles, normalize=False)
                            for i, f in enumerate(reduced_graph.node_features):
                                if len(args.reduced) == 0:
                                    f_tuple = getImportanceFeatures(['atom'], f)
                                elif r != 'substructure':
                                    f_tuple = getImportanceFeatures([r], f)
                                else:
                                    f_tuple = getImportanceFeatures([r], reduced_graph.cliques_smiles[i])
                                
                                # Use .get() for safe dictionary access with warning for missing keys
                                if i not in mask_graph_r['atom']:
                                    warnings.warn(f"Reduced graph index {i} not found in mask_graph_r. Using default weight 0.0.")
                                weight = mask_graph_r['atom'].get(i, 0.0)
                                molattention.append(MolAttention(r, i, f_tuple, weight))

                    if not args.schema in ['A']:
                        mask_graph_x = mask_rtog(smiles, cliques, mask_graph_r)
                        if args.schema in ['AR', 'AR_0', 'AR_N']:
                            mask_graph_x = mask_gandr(mask_graph_g, mask_graph_x)
                            reduced_graph, cliques, edges = getReducedGraph(args, ['atom'], smiles, normalize=False)
                            for i, f in enumerate(reduced_graph.node_features):
                                f_tuple = getImportanceFeatures(['atom'], f)
                                # Use .get() for safe dictionary access with warning for missing keys
                                if i not in mask_graph_x['atom']:
                                    warnings.warn(f"Atom index {i} not found in mask_graph_x. Using default weight 0.0.")
                                weight = mask_graph_x['atom'].get(i, 0.0)
                                molattention.append(MolAttention('x', i, f_tuple, weight))

                    molprediction.append(MolPrediction(fold_number, smiles, spilttingdataset, predicted, truevalue, molattention))
                except Exception as e:
                    print(f"Error processing molecule {smiles} in fold {fold_number}: {e}")
                    continue

    if len(molprediction) == 0:
        warnings.warn("No molecules were successfully processed. Returning empty DataFrame.")
        import pandas as pd
        return pd.DataFrame(columns=['Fold', 'SMILES_original', 'SMILES', 'Split', 'Prediction', 'TrueValue', 'Schema', 'Node_ID', 'Node_Feature', 'Weight', 'Weight_Other'])
    
    prediction_fold_df = pd.concat([p.to_dataframe() for p in molprediction], axis=0)
    return prediction_fold_df

def getPredictionConsensus(prediction_fold_df):
    groupby_column = ['SMILES_original', 'SMILES', 'Split', 'Schema', 'Node_ID', 'Node_Feature']
    selected_column = ['Prediction', 'TrueValue', 'Weight', 'Weight_Other']
    prediction_consensus_df = prediction_fold_df.groupby(groupby_column).mean()[selected_column]
    prediction_consensus_df.reset_index(inplace=True)
    prediction_consensus_df['Fold'] = 'C'
    prediction_consensus_df = prediction_consensus_df[prediction_fold_df.columns]
    return prediction_consensus_df

def getPrediction(prediction_mol_df, args, print_result=False):
    prediction = prediction_mol_df['Prediction'].iloc[0]
    if print_result: print('Prediction Result (class):', set(prediction_mol_df['Prediction']))
    if args.graphtask == 'classification':
        if print_result: print('Prediction Result (class):', float(list(set(prediction_mol_df['Prediction']))[0])>0.5)

    smiles = prediction_mol_df['SMILES'].iloc[0]
    mol = Chem.MolFromSmiles(smiles)
    graph_g_df = prediction_mol_df[prediction_mol_df['Schema']=='atom']
    graph_r_df = prediction_mol_df[(prediction_mol_df['Schema']!='atom') & (prediction_mol_df['Schema']!='x')]
    graph_x_df = prediction_mol_df[prediction_mol_df['Schema']=='x']

    graph_g = {'atom': {}, 'bond': {}}
    for i, row in graph_g_df.iterrows():
        graph_g['atom'][row['Node_ID']] = row['Weight']
    if len(graph_g_df)>0: graph_g = minmaxnormalize(graph_g)
    graph_r = {'atom': {}, 'bond': {}}
    for i, row in graph_r_df.iterrows():
        graph_r['atom'][row['Node_ID']] = row['Weight']
    if len(graph_r_df)>0: graph_r = minmaxnormalize(graph_r) 
    graph_x = {'atom': {}, 'bond': {}}
    for i, row in graph_x_df.iterrows():
        graph_x['atom'][row['Node_ID']] = row['Weight']
    if len(graph_x_df)>0: graph_x = minmaxnormalize(graph_x)

    if len(args.reduced) >= 1:
        r = list(set(prediction_mol_df['Schema'])-set(['atom', 'x']))[0]
        reduced_graph, cliques, edges = getReducedGraph(args, [r], smiles, normalize=False)

    if print_result: print('graph_g', graph_g)
    if print_result: display_interpret_weight(mol, None, None, graph_g, None, scale=True)
    if len(args.reduced) >= 1:
        if print_result: print('graph_r', graph_r)
        if print_result: display_interpret_weight(mol, cliques, edges, graph_g, graph_r, scale=True)
    if print_result: print('graph_x', graph_x)
    if print_result: display_interpret_weight(mol, None, None, graph_x, None, scale=True)

    return {'prediction': prediction, 
            'graph_g': graph_g, 
            'graph_r': graph_r, 
            'graph_x': graph_x}
    

def getSubstructureFold(args, args_test, all_dataset, datasets_splitted=None, print_result=False):

    fragment_dict = dict()
    no_fragment = list()
    path = 'dataset/'+args.file+'/fragment.pickle'

    # if os.path.exists(path):
    #     with open(path, 'rb') as handle:
    #         fragment_dict = pickle.load(handle)
    # else:
    #     for g in tqdm(all_dataset):
    #         mol = smiles_to_mol(g, with_atom_index=False, kekulize=False)
    #         limit = [3,20] # 3-20 atoms
    #         fragment_recap = recap_frag_smiles_children([mol], limit=limit)
    #         fragment_brics = brics_frag_smiles([mol], limit=limit)
    #         fragment_grinder = grinder_frag_smiles([mol], limit=limit)
    #         fragment_all_t = fragment_recap.union(fragment_brics)
    #         fragment_all = fragment_all_t.union(fragment_grinder)
    #         fragment_dict[g] = fragment_all
    #         if len(fragment_all) == 0:
    #             no_fragment.append(g)
    #     # write dataset to pickle 
    #     with open(path, 'wb') as handle:
    #         pickle.dump(fragment_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    molprediction = []

    # loop all fold
    for fold_number in range(args.fold):
    # for fold_number in tqdm([4]):
        args_test['fold_number'] = fold_number
        tester = Tester(args, args_test, print_model=False)

        for d in tqdm(all_dataset, desc='getMaskGraph'):
            # data loader
            test_loader, datasets_test =  generateDataLoaderListing([all_dataset[d]], 1)
            molecule_test = datasets_test[0]

            # molecule
            sample_graph = molecule_test
            smiles = sample_graph.smiles
            mol = Chem.MolFromSmiles(smiles)

            molattentionsubstructure = []
            
            # testing
            try:
                spilttingdataset = 'test' if molecule_test.smiles in list(datasets_splitted[1]['X']) else 'train'
            except:
                spilttingdataset = '-'

            # true value
            truevalue = molecule_test.y
            try:
                truevalue = molecule_test.y.item()
            except:
                truevalue = molecule_test.y[0][0]

            # testing
            # print(datasets_test)
            try:
                predicted = tester.test_single(test_loader, return_attention_weights=True, print_result=False, raw_prediction=True)
                try:
                    predicted = predicted.item()
                except:
                    predicted = predicted[0][0]
            except Exception as e:
                print(e)
                predicted = None

            # print(predicted)
            # if predicted != molecule_test.y:
            #     print(molecule_test.smiles, "TRUE:", molecule_test.y, "PREDICTED:", predicted)

            if predicted is not None:

                try:
                    # attention result
                    att_mol = tester.getAttentionMol()
                    sample_att = att_mol
                    if 'atom' in sample_att:
                        sample_att_g = sample_att['atom']
                    else:
                        sample_att_g = None
                    if len(args.reduced) >= 1:
                        # Use the first reduced graph for mask_graph_r computation
                        reduced_key = args.reduced[0]
                        if reduced_key in sample_att:
                            sample_att_r = sample_att[reduced_key]
                        else:
                            # Try to find any available reduced graph key
                            available_keys = [k for k in sample_att.keys() if k != 'atom']
                            if available_keys:
                                sample_att_r = sample_att[available_keys[0]]
                                warnings.warn(f"Reduced graph '{reduced_key}' not found in attention. Using '{available_keys[0]}' instead.")
                            else:
                                sample_att_r = None
                                warnings.warn(f"No reduced graph attention found. Skipping reduced graph processing.")
                    else:
                        sample_att_r = None
                    # sample_att_g, sample_att_r = sample_att
                    if args.schema in ['A', 'R_N', 'AR', 'AR_0', 'AR_N']:
                        mask_graph_g = mask_graph(sample_att_g)
                    if args.schema in ['R', 'R_0', 'R_N', 'AR', 'AR_0', 'AR_N']:
                        mask_graph_r = mask_reduced(sample_att_r)
                    
                    mask_graph_x = None

                    # atom graph
                    if 'A' in args.schema:
                        mask_graph_x = mask_graph_g
                        # reduced_graph, cliques, edges = getReducedGraph(args, ['atom'], smiles, normalize=False)

                    # reduced graph
                    if 'R' in args.schema:
                        # mask_graph_x = mask_graph_r
                        for r in args.reduced:
                            reduced_graph, cliques, edges = getReducedGraph(args, [r], smiles, normalize=False)

                    if not args.schema in ['A']:
                        mask_graph_x = mask_rtog(smiles, cliques, mask_graph_r)
                        if args.schema in ['AR', 'AR_0', 'AR_N']:
                            mask_graph_x = mask_gandr(mask_graph_g, mask_graph_x)
                            # reduced_graph, cliques, edges = getReducedGraph(args, ['atom'], smiles, normalize=False)
                        
                    # get series of reasonable fragments
                    # print(smiles)
                    if smiles in fragment_dict:
                        fragment_all = fragment_dict[smiles]
                    else:
                        limit = [3,20] # 3-20 atoms
                        # fragment_recap = recap_frag_smiles_children([mol], limit=limit)
                        fragment_brics = brics_frag_smiles([mol], limit=limit)
                        # fragment_grinder = grinder_frag_smiles([mol], limit=limit)
                        # fragment_all_t = fragment_recap.union(fragment_brics)
                        # fragment_all = fragment_all_t.union(fragment_grinder)
                        fragment_all = fragment_brics

                    for frag in fragment_all:

                        frag_smarts = frag
                        frag_mol = Chem.MolFromSmarts(frag_smarts)
                        if frag_mol is None:
                            warnings.warn(f"Failed to parse SMARTS pattern: {frag_smarts}. Skipping this fragment.")
                            continue

                        frag_match = mol.GetSubstructMatches(frag_mol)

                        for ff in frag_match:
                            attention_frag = list()
                            attention_nonfrag = list()
                            for i in mask_graph_x['atom']:
                                if i in ff:
                                    attention_frag.append(mask_graph_x['atom'][i])
                                else:
                                    attention_nonfrag.append(mask_graph_x['atom'][i])
                            # Validate fragment matching with safe checks
                            if len(attention_frag) != len(ff):
                                warnings.warn(f"Fragment length mismatch for {frag_smarts}: expected {len(ff)}, got {len(attention_frag)}. Skipping.")
                                continue
                            if len(attention_frag)+len(attention_nonfrag) != len(mask_graph_x['atom']):
                                warnings.warn(f"Total attention count mismatch for {frag_smarts}. Skipping.")
                                continue
                            molattentionsubstructure.append(MolAttentionSubstructure('x', i, frag_smarts, attention_frag, attention_nonfrag))

                    molprediction.append(MolPrediction(fold_number, smiles, spilttingdataset, predicted, truevalue, molattentionsubstructure))
                except Exception as e:
                    print(f"Error processing molecule {smiles} in fold {fold_number}: {e}")
                    continue

    if len(molprediction) == 0:
        warnings.warn("No molecules were successfully processed. Returning empty DataFrame.")
        import pandas as pd
        return pd.DataFrame(columns=['Fold', 'SMILES_original', 'SMILES', 'Split', 'Prediction', 'TrueValue', 'Schema', 'Node_ID', 'Node_Feature', 'Weight', 'Weight_Other'])
    
    prediction_fold_df = pd.concat([p.to_dataframe() for p in molprediction], axis=0)
    return prediction_fold_df
    
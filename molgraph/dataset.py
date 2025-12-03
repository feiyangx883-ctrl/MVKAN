######################
### Import Library ###
######################

# my library
from molgraph.molgraph import *
from molgraph.substructurevocab import *
from molgraph.utilsmol import *
# standard
import os as os
import json
import re
import pandas as pd
import numpy as np
import pickle as pickle
import csv as csv
from tqdm import tqdm
# pytorch
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# deepchem
import deepchem as dc



# ----------------------------
# Manual label specifications
# ----------------------------


def get_sider():
    """Return the ordered SIDER side-effect categories."""

    # sider,27
    return [
        'Hepatobiliary disorders',
        'Metabolism and nutrition disorders',
        'Product issues',
        'Eye disorders',
        'Investigations',
        'Musculoskeletal and connective tissue disorders',
        'Gastrointestinal disorders',
        'Social circumstances',
        'Immune system disorders',
        'Reproductive system and breast disorders',
        'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
        'General disorders and administration site conditions',
        'Endocrine disorders',
        'Surgical and medical procedures',
        'Vascular disorders',
        'Blood and lymphatic system disorders',
        'Skin and subcutaneous tissue disorders',
        'Congenital, familial and genetic disorders',
        'Infections and infestations',
        'Respiratory, thoracic and mediastinal disorders',
        'Psychiatric disorders',
        'Renal and urinary disorders',
        'Pregnancy, puerperium and perinatal conditions',
        'Ear and labyrinth disorders',
        'Cardiac disorders',
        'Nervous system disorders',
        'Injury, poisoning and procedural complications',
    ]


_MANUAL_MULTITASK_LABELS = {
    'sider': get_sider(),
}


def _get_manual_multitask_labels(dataset_name):
    if dataset_name is None:
        return None
    return _MANUAL_MULTITASK_LABELS.get(str(dataset_name).lower())


def _normalize_label_key(name):
    return re.sub(r'[^a-z0-9]+', '', str(name).lower())


def _match_manual_labels(source_df, manual_labels):
    """Map manually recorded labels to actual DataFrame columns."""

    normalized_columns = {
        _normalize_label_key(col): col for col in source_df.columns
    }
    matched_columns = []
    for label in manual_labels:
        key = _normalize_label_key(label)
        column_name = normalized_columns.get(key)
        if column_name is None:
            return None
        matched_columns.append(column_name)
    return matched_columns

########################
### Dataset Function ###
########################


# write disk dataset to CSV file
def _is_sequence_label(value):
    """Return True if *value* behaves like a sequence of targets."""
    return isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, (str, bytes))


def _coerce_label_value(value):
    """Best-effort conversion of CSV cell contents into scalar or sequence labels."""
    if _is_sequence_label(value):
        return value
    if isinstance(value, str):
        value_strip = value.strip()
        if value_strip.startswith('[') and value_strip.endswith(']'):
            try:
                loaded = json.loads(value_strip)
                if _is_sequence_label(loaded):
                    return loaded
            except json.JSONDecodeError:
                pass
        try:
            return float(value_strip)
        except ValueError:
            return value
    return value


def _labels_to_dataframe_column(labels):
    """Expand a sequence of labels into a DataFrame with y_0, y_1, ... columns."""
    label_array = [np.array(v, dtype=float).tolist() for v in labels]
    label_df = pd.DataFrame(label_array)
    label_df.columns = [f'y_{i}' for i in range(label_df.shape[1])]
    return label_df


def _make_hashable_label(value):
    """Convert label containers into hashable tuples for grouping operations."""

    if _is_sequence_label(value):
        return tuple(np.asarray(value, dtype=float).tolist())
    return value


def _reconstruct_labels(df):
    """Recreate the aggregated 'y' column from expanded y_* columns if required."""
    y_cols = [c for c in df.columns if c.startswith('y_')]
    if not y_cols:
        return df

    label_matrix = df[y_cols].to_numpy(dtype=float)
    if label_matrix.shape[1] == 1:
        df['y'] = label_matrix.ravel()
    else:
        df['y'] = [row.tolist() for row in label_matrix]
    df = df.drop(columns=y_cols)
    return df

# write disk dataset to CSV file
def writeToCSV(datasets, path):
    df = datasets.copy()
    if 'y' in df.columns and df['y'].apply(_is_sequence_label).any():
        label_df = _labels_to_dataframe_column(df['y'])
        df = pd.concat([df.drop(columns=['y']), label_df], axis=1)
    df.to_csv(path, index=False)

# write disk dataset to CSV file
def readFromCSV(path):
    df = pd.read_csv(path)
    df = _reconstruct_labels(df)
    if 'y' in df.columns:
        df['y'] = df['y'].apply(_coerce_label_value)
    return df

# Dataset getting and construction dataframe
def getDatasetFromFile(file, smiles, task, splitting):
    """
    从文件加载数据集并准备训练数据
    
    Args:
        file: 数据集名称
        smiles: SMILES 列名
        task: 任务列名，可以是:
              - 'auto': 自动检测任务列
              - 单个列名 (字符串): 单任务
              - 多个列名 (用'|'分隔的字符串或列表): 多任务
        splitting: 分割策略
        
    Returns:
        处理后的数据集 DataFrame
    """
    path = f'dataset/{file}.csv'

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # 读取数据集
    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    
    print(f'\n=== Loading Dataset: {file} ===')
    print(f'Available columns: {df.columns.tolist()}')
    print(f'Dataset shape: {df.shape}')
    
    if smiles not in df.columns:
        raise KeyError(
            f"SMILES column '{smiles}' not found in dataset '{file}'.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    
    # 1. 处理 'auto' 自动检测
    if task == 'auto' or task is None:
        print(f"Auto-detecting task columns...")
        
        # 首先尝试使用手动定义的标签（针对已知的多任务数据集）
        manual_labels = _get_manual_multitask_labels(file)
        if manual_labels:
            matched_cols = _match_manual_labels(df, manual_labels)
            if matched_cols:
                task_cols = matched_cols
                print(f"✓ Using manually defined {len(task_cols)} tasks for '{file}'")
            else:
                print(f"⚠ Manual labels defined but not matched, falling back to auto-detection")
                task_cols = None
        else:
            task_cols = None
        
        # 如果没有手动定义或匹配失败，自动检测
        if task_cols is None:
            # 排除 SMILES 列和其他非标签列
            exclude_cols = {
                smiles.lower(), 
                'smiles', 'mol', 'molecule', 
                'id', 'ids', 'name', 'names',
                'split', 'subset', 'fold'
            }
            exclude_cols.update({
                c for c in df.columns 
                if str(c).lower().startswith('unnamed')
            })
            
            # 查找潜在的标签列（数值型）
            potential_tasks = []
            for col in df.columns:
                if col.lower() in exclude_cols or col == smiles:
                    continue
                # 检查是否为数值型列
                if pd.api.types.is_numeric_dtype(df[col]):
                    potential_tasks.append(col)
            
            if not potential_tasks:
                raise ValueError(
                    f"No numeric task columns found in dataset '{file}'.\n"
                    f"Available columns: {df.columns.tolist()}\n"
                    f"Excluded: {exclude_cols}"
                )
            
            task_cols = potential_tasks
            print(f"✓ Auto-detected {len(task_cols)} numeric task(s): {task_cols[:5]}{'...' if len(task_cols) > 5 else ''}")
    
    # 2. 处理用 '|' 分隔的任务字符串
    elif isinstance(task, str) and '|' in task:
        task_cols = [t.strip() for t in task.split('|')]
        print(f"Using {len(task_cols)} specified tasks (separated by '|')")
    
    # 3. 处理列表
    elif isinstance(task, (list, tuple)):
        task_cols = [str(t).strip() for t in task]
        print(f"Using {len(task_cols)} specified tasks (from list)")
    
    # 4. 处理单个字符串（单任务）
    elif isinstance(task, str):
        task_cols = [task.strip()]
        print(f"Using single task: '{task_cols[0]}'")
    
    else:
        raise TypeError(f"Invalid task type: {type(task)}, value: {task}")

    # ===== 验证任务列是否存在 =====
    missing_cols = [col for col in task_cols if col not in df.columns]
    if missing_cols:
        # 尝试大小写不敏感匹配
        lower_map = {c.lower(): c for c in df.columns}
        remapped_cols = []
        still_missing = []
        
        for col in task_cols:
            lowered = col.lower()
            if lowered in lower_map:
                remapped_cols.append(lower_map[lowered])
            else:
                still_missing.append(col)
        
        if still_missing:
            raise KeyError(
                f"Task columns {still_missing} not found in dataset '{file}'.\n"
                f"Available columns: {df.columns.tolist()}\n"
                f"Requested tasks: {task_cols}"
            )
        
        task_cols = remapped_cols
        print(f"⚠ Applied case-insensitive column matching")

    # ===== 提取标签数据 =====
    def _extract_multitask_labels(source_df, expected_cols):
        """提取多任务标签的通用函数"""
        # 直接从列中提取
        available = [c for c in expected_cols if c in source_df.columns]
        if len(available) == len(expected_cols):
            label_values = source_df[available].to_numpy(dtype=float)
            return [row.tolist() for row in label_values]
        
        # 如果有聚合的 'y' 列
        if 'y' in source_df.columns:
            label_series = source_df['y'].apply(_coerce_label_value)
            if label_series.apply(_is_sequence_label).all():
                labels = []
                for idx, value in enumerate(label_series):
                    arr = np.array(value, dtype=float)
                    if arr.ndim == 0:
                        raise ValueError(
                            f"Row {idx} in 'y' column is scalar but multi-task expected"
                        )
                    labels.append(arr.tolist())
                return labels
        
        raise KeyError(f"Could not extract labels from columns: {expected_cols}")

    # 提取标签
    if len(task_cols) == 1:
        # 单任务
        label_col = task_cols[0]
        if label_col in df.columns:
            label_series = df[label_col]
        elif 'y' in df.columns:
            label_series = df['y']
        else:
            raise KeyError(f"Column '{label_col}' not found in dataset '{file}'")
        
        label_series = label_series.apply(_coerce_label_value)
        df_new = pd.DataFrame({'X': df[smiles], 'y': label_series})
        print(f"✓ Single-task dataset created")
    else:
        # 多任务
        labels = _extract_multitask_labels(df, task_cols)
        df_new = pd.DataFrame({'X': df[smiles], 'y': labels})
        print(f"✓ Multi-task dataset created with {len(task_cols)} tasks")

    # 添加其他必要的列
    df_new['ids'] = list(df[smiles])  # 原始 SMILES
    
    # 处理分割信息
    if splitting not in ['random', 'scaffold'] and splitting in df.columns:
        df_new['s'] = df[splitting]
        print(f"✓ Using predefined split from column '{splitting}'")
    else:
        df_new['s'] = np.zeros(len(df_new))
        print(f"✓ Will use {splitting} splitting strategy")

    datasets = df_new
    
    print(f'✓ Dataset loaded successfully')
    print(f'  Total samples: {len(datasets)}')
    print(f'  SMILES column: {smiles}')
    print(f'  Task columns: {task_cols}')
    print('='*50 + '\n')

    return datasets


# Recheck all valid smiles in dataset
def getValidDataset(datasets):
    not_sucessful = 0
    single_atom = 0
    processed_smiles = 0
    X = datasets['X']
    
    valid_smiles = list()
    validated = list()
    for smiles in tqdm(X.values, desc='===Checking valid smiles==='):
        try:
            smiles_valid = getValidSmiles(smiles)
            # not included single atom
            if checkSingleAtom(smiles_valid):
                single_atom += 1
                valid_smiles.append(smiles)
                validated.append('invalid')
                continue
            processed_smiles += int(smiles_valid != smiles)
            valid_smiles.append(smiles_valid)
            validated.append('valid')
        except:
            not_sucessful += 1
            valid_smiles.append(smiles)
            validated.append('invalid')
            pass
    
    datasets['X'] = valid_smiles
    datasets['v'] = validated
    print('Function: getValidDataset()')
    print("number of all smiles:", len(list(X)))
    print("number of valid smiles:", validated.count('valid'))
    print("number of failed smiles (rdkit):", not_sucessful)
    print("number of failed smiles (single atom):", single_atom)
    print("number of processed smiles:", processed_smiles)

    new_datasets = datasets
    return new_datasets

# remove conflict smiles (Same X diff y)
def getNonConflictDataset(datasets):
    # result = datasets.groupby('X')['y'].transform('nunique') > 1
    if 'y' not in datasets.columns:
        return datasets

    y_hashable = datasets['y'].apply(_make_hashable_label)
    result = y_hashable.groupby(datasets['X']).transform('nunique') > 1
    dataset_conflict = datasets[result]
    print("number of conflict smiles:", len(list(dataset_conflict['X'])))
    new_datasets = datasets
    new_datasets.loc[result, 'v'] = 'invalid'
    print("number of valid smiles:", list(datasets['v']).count('valid'))
    return new_datasets

# get validated dataset 
def getDataset(file, smiles, task, splitting):
    # creating file folder
    if not(os.path.isdir('./dataset/{}'.format(file))):
        os.makedirs(os.path.join('./dataset/{}'.format(file)))
    # read validated if exists
    path = 'dataset/'+file+'/'+file+'_validated.csv'
    if os.path.exists(path):
        datasets = readFromCSV(path)
        datasets = datasets[datasets['v']=='valid']
    else:
        # retrieve data from file
        datasets = getDatasetFromFile(file, smiles, task, splitting)
        # validate smiles
        datasets = getValidDataset(datasets)
        # remove conflict smiles (Same X diff y)
        datasets = getNonConflictDataset(datasets)
        # write to files
        writeToCSV(datasets, path)  

    datasets = datasets[datasets['v']=='valid']
    print('Function: getDataset()')
    print("number of valid smiles:", len(list(datasets['X'])))
    return datasets

def getPosWeight(datasets):
    """
    计算正样本权重
    - 单任务: 返回标量
    - 多任务: 返回 numpy 数组,每个任务一个权重
    """
    y_values = list(datasets['y'].values)
    if len(y_values) == 0:
        return 1.0

    first_value = y_values[0]
    if _is_sequence_label(first_value):
        # 多任务场景
        y_matrix = np.array([np.array(v, dtype=float) for v in y_values])
        
        # 处理 NaN 值 (某些任务可能有缺失标签)
        pos_weights = []
        for task_idx in range(y_matrix.shape[1]):
            task_labels = y_matrix[:, task_idx]
            # 过滤掉 NaN
            valid_mask = ~np.isnan(task_labels)
            valid_labels = task_labels[valid_mask]
            
            if len(valid_labels) == 0:
                # 所有标签都是 NaN
                pos_weights.append(1.0)
            else:
                pos_count = np.sum(valid_labels == 1)
                neg_count = np.sum(valid_labels == 0)
                
                if pos_count == 0:
                    # 没有正样本,使用默认权重
                    pos_weights.append(1.0)
                else:
                    # neg/pos 比例
                    weight = neg_count / pos_count
                    pos_weights.append(weight)
        
        pos_weights_array = np.array(pos_weights)
        
        # 打印统计信息
        print(f'\n多任务权重统计:')
        print(f'  任务数量: {len(pos_weights_array)}')
        print(f'  权重范围: [{pos_weights_array.min():.2f}, {pos_weights_array.max():.2f}]')
        print(f'  平均权重: {pos_weights_array.mean():.2f}')
        
        return pos_weights_array
    else:
        # 单任务场景
        y_array = np.array(y_values, dtype=float)
        pos_count = np.sum(y_array == 1)
        neg_count = np.sum(y_array == 0)
        pos_count = pos_count if pos_count != 0 else 1.0
        weight = neg_count / pos_count
        
        print(f'\n单任务权重统计:')
        print(f'  正样本: {int(pos_count)}, 负样本: {int(neg_count)}')
        print(f'  pos_weight: {weight:.4f}')
        
        return weight


# generate dataset with splitting
def generateDatasetSplitting(file, splitting=None, splitting_fold=None, splitting_seed=None):
    path = 'dataset/'+file+'/train_'+str(splitting_seed)+'_'+str(splitting_fold-1)+'.csv'
    
    if os.path.exists(path):
        datasets_fold = []
        for i in range(splitting_fold):
            path_train = 'dataset/'+file+'/train_'+str(splitting_seed)+'_'+str(i)+'.csv'
            path_val = 'dataset/'+file+'/val_'+str(splitting_seed)+'_'+str(i)+'.csv'
            datasets_fold.append((readFromCSV(path_train), readFromCSV(path_val)))
        path_test = 'dataset/'+file+'/test.csv'
        datasets_test = readFromCSV(path_test)
    else:
        # read validated if exists
        path = 'dataset/'+file+'/'+file+'_validated.csv'
        if os.path.exists(path):
            datasets = readFromCSV(path)
            datasets = datasets[datasets['v']=='valid']
            # datasets = datasets.sample(frac=1) # in case shuffle is needed
        
        # construct splitting method
        if splitting == 'random':
            print(splitting, 'RandomSplitter')
            # convert to DiskDataset using smiles as ids (original smiles)
            y_array = np.array(list(datasets['y'].values), dtype=object)
            if y_array.ndim == 1 and _is_sequence_label(y_array[0]):
                y_array = np.vstack([np.array(v, dtype=float) for v in y_array])
            else:
                y_array = np.array(y_array, dtype=float).reshape(-1, 1)
            datasets = dc.data.DiskDataset.from_numpy(X=datasets['X'].values,
                                                     y=y_array,
                                                     ids=datasets['ids'].values)
            splitter_s = dc.splits.RandomSplitter()
            # return Numpy Dataset object x, y, w, ids
            if splitting_fold != 1:
                datasets_trainval, datasets_test = splitter_s.train_test_split(dataset=datasets, seed=splitting_seed)
                datasets_fold = splitter_s.k_fold_split(dataset=datasets_trainval, k=splitting_fold)
            else: # no cross validation
                datasets_train, datasets_val, datasets_test = splitter_s.train_valid_test_split(dataset=datasets, seed=splitting_seed)
                datasets_fold = [(datasets_train, datasets_val)]

        elif splitting == 'scaffold':
            print(splitting, 'ScaffoldSplitter')
            # convert to DiskDataset using smiles as ids (original smiles)
            y_array = np.array(list(datasets['y'].values), dtype=object)
            if y_array.ndim == 1 and _is_sequence_label(y_array[0]):
                y_array = np.vstack([np.array(v, dtype=float) for v in y_array])
            else:
                y_array = np.array(y_array, dtype=float).reshape(-1, 1)
            datasets = dc.data.DiskDataset.from_numpy(X=datasets['X'].values,
                                                     y=y_array,
                                                     ids=datasets['ids'].values)
            splitter_s = dc.splits.ScaffoldSplitter()
            # return Numpy Dataset object x, y, w, ids
            if splitting_fold != 1:
                datasets_trainval, datasets_test = splitter_s.train_test_split(dataset=datasets, seed=splitting_seed)
                datasets_fold = splitter_s.k_fold_split(dataset=datasets_trainval, k=splitting_fold)
            else: # no cross validation
                datasets_train, datasets_val, datasets_test = splitter_s.train_valid_test_split(dataset=datasets, seed=splitting_seed)
                datasets_fold = [(datasets_train, datasets_val)]

        else:
            print(splitting, 'Defined')
            datasets_trainval_X = list()
            datasets_trainval_y = list()
            datasets_trainval_ids = list()
            datasets_test_X = list()
            datasets_test_y = list()
            datasets_test_ids = list()
            for idx, row in datasets.iterrows():
                if row['s'].lower() != 'test':
                    datasets_trainval_X.append(row['X'])
                    datasets_trainval_y.append(row['y'])
                    datasets_trainval_ids.append(row['ids'])
                elif row['s'].lower() == 'test':
                    datasets_test_X.append(row['X'])
                    datasets_test_y.append(row['y'])
                    datasets_test_ids.append(row['ids'])

            y_trainval = np.array(datasets_trainval_y, dtype=object)
            if y_trainval.ndim == 1 and _is_sequence_label(y_trainval[0]):
                y_trainval = np.vstack([np.array(v, dtype=float) for v in y_trainval])
            else:
                y_trainval = np.array(y_trainval, dtype=float).reshape(-1, 1)

            y_test = np.array(datasets_test_y, dtype=object)
            if y_test.ndim == 1 and _is_sequence_label(y_test[0]):
                y_test = np.vstack([np.array(v, dtype=float) for v in y_test])
            else:
                y_test = np.array(y_test, dtype=float).reshape(-1, 1)

            datasets_trainval = dc.data.DiskDataset.from_numpy(X=datasets_trainval_X,
                                                               y=y_trainval,
                                                               ids=datasets_trainval_ids)
            datasets_test = dc.data.DiskDataset.from_numpy(X=datasets_test_X,
                                                           y=y_test,
                                                           ids=datasets_test_ids)
            # return Numpy Dataset object x, y, w, ids
            splitter_s = dc.splits.RandomSplitter()
            if splitting_fold != 1:
                datasets_fold = splitter_s.k_fold_split(dataset=datasets_trainval, k=splitting_fold)
            else:
                datasets_train, datasets_val = splitter_s.train_test_split(dataset=datasets_trainval, frac_train=0.9, seed=splitting_seed)
                datasets_fold = [(datasets_train, datasets_val)]
                
        datasets_fold_df = list()
        # write dataset to csv
        for i in range(len(datasets_fold)):
            train_df = datasets_fold[i][0].to_dataframe()
            train_df = _reconstruct_labels(train_df)
            val_df = datasets_fold[i][1].to_dataframe()
            val_df = _reconstruct_labels(val_df)
            writeToCSV(train_df, 'dataset/'+file+'/train_'+str(splitting_seed)+'_'+str(i)+'.csv')
            writeToCSV(val_df, 'dataset/'+file+'/val_'+str(splitting_seed)+'_'+str(i)+'.csv')
            datasets_fold_df.append((train_df, val_df))
        test_df = datasets_test.to_dataframe()
        test_df = _reconstruct_labels(test_df)
        writeToCSV(test_df, 'dataset/'+file+'/test.csv')
        datasets_fold = datasets_fold_df
        datasets_test = test_df

    datasets_splitted = (datasets_fold, datasets_test) 
    print('Function: generateDatasetSplitting()')
    print('Fold:', len(datasets_fold))
    for i, f in enumerate(datasets_fold):
        print('Fold Number:', i)
        print('-- Datasets Train:', len(list(f[0]['X'])))
        print('-- Datasets Val:', len(list(f[1]['X'])))
        print('-- Datasets Test: ', len(datasets_test['X']))
        print('-- Total:', len(list(f[0]['X']))+len(list(f[1]['X']))+len(datasets_test['X']))

    return datasets_splitted

##################
### Data Class ###
##################

# Pair data class
class PairData(Data):
    def __init__(self, edge_index_g=None, x_g=None, edge_index_r=None, x_r=None, **kwargs):
        super().__init__()
        
        if x_g is not None:
            self.x_g = x_g
        if edge_index_g is not None:
            self.edge_index_g = edge_index_g
        if x_r is not None:
            self.x_r = x_r
        if edge_index_r is not None:
            self.edge_index_r = edge_index_r
            
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_g':
            return self.x_g.size(0)
        elif key == 'edge_index_r':
            return self.x_r.size(0)
        elif key == 'pooling_index_g':
            return self.num_nodes_g.item()
        elif key == 'pooling_index_r':
            return self.num_nodes_r.item()
        else:
            return super().__inc__(key, value, *args, **kwargs)


# Generate Pair data for atom-graph
def constructGraph(smiles, y, ids=None):
    try:
        # pair data
        d = PairData()
        d.smiles = smiles
        if ids: d.ids = ids
        atom_graph = AtomGraph(smiles)
        graph_size, node_index, node_features, edge_index, edge_features = atom_graph.graph_size, atom_graph.node_index, atom_graph.node_features, atom_graph.edge_index, atom_graph.edge_features

        # feature values
        d.x_g = torch.tensor(np.array(node_features, dtype=float)).type(torch.DoubleTensor)
        if graph_size == 1:
            print('WARNING (SINGLE ATOM):', smiles)
            edge_index = [[],[]]
            d.edge_index_g = torch.tensor(np.array(edge_index))
            d.edge_attr_g = torch.Tensor()
        else:
            if len(edge_index) == 0:
                d.edge_index_g = torch.tensor(np.array([[],[]])).type(torch.LongTensor)
            else:
                d.edge_index_g = torch.transpose(torch.tensor(edge_index), 0, 1)
            if len(edge_features) == 0:
                d.edge_attr_g = torch.Tensor(np.array([]))
            else:
                d.edge_attr_g = torch.tensor(np.array(edge_features, dtype=float)).type(torch.DoubleTensor)
        # predicting value
        y_array = np.array(y, dtype=float)
        if y_array.ndim == 0:
            y_array = np.array([[y_array]])
        else:
            y_array = y_array.reshape(1, -1)
        d.y = torch.tensor(y_array).type(torch.DoubleTensor)
        return d

    except Exception as e:
        print('ERROR (MOL FAIL):', smiles, e)
        return None

# Generate Pair data for reduced-graph
def constructReducedGraph(reducedgraph, smiles, y, tokenizer=None):
    try:
        # pair data
        d = PairData()
        d.smiles = smiles
    
        # reduced graph
        if reducedgraph == 'atom':
            atom_graph = AtomGraph(smiles)
            cliques = atom_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = atom_graph.graph_size, atom_graph.node_index, atom_graph.node_features, atom_graph.edge_index, atom_graph.edge_features
        elif reducedgraph == 'junctiontree':
            junctiontree_graph = JunctionTreeGraph(smiles)
            cliques = junctiontree_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = junctiontree_graph.graph_size, junctiontree_graph.node_index, junctiontree_graph.node_features, junctiontree_graph.edge_index, junctiontree_graph.edge_features
        elif reducedgraph == 'cluster':
            cluster_graph = ClusterGraph(smiles)
            cliques = cluster_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = cluster_graph.graph_size, cluster_graph.node_index, cluster_graph.node_features, cluster_graph.edge_index, cluster_graph.edge_features
        elif reducedgraph == 'functional':
            functional_graph = FunctionalGraph(smiles)
            cliques = functional_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = functional_graph.graph_size, functional_graph.node_index, functional_graph.node_features, functional_graph.edge_index, functional_graph.edge_features
        elif reducedgraph == 'pharmacophore':
            pharmacophore_graph = PharmacophoreGraph(smiles)
            cliques = pharmacophore_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = pharmacophore_graph.graph_size, pharmacophore_graph.node_index, pharmacophore_graph.node_features, pharmacophore_graph.edge_index, pharmacophore_graph.edge_features
        elif reducedgraph == 'substructure':
            substructure_graph = SubstructureGraph(smiles, tokenizer)
            cliques = substructure_graph.cliques
            graph_size, node_index, node_features, edge_index, edge_features = substructure_graph.graph_size, substructure_graph.node_index, substructure_graph.node_features, substructure_graph.edge_index, substructure_graph.edge_features

        # feature values
        d.x_r = torch.tensor(np.array(node_features, dtype=float)).type(torch.DoubleTensor)
        if graph_size > 1:
            if len(edge_index) == 0:
                d.edge_index_r = torch.tensor(np.array([[],[]])).type(torch.LongTensor)
            else:
                d.edge_index_r = torch.transpose(torch.tensor(edge_index), 0, 1)
            if len(edge_features) == 0:
                d.edge_attr_r = torch.Tensor(np.array([]))
            else:
                d.edge_attr_r = torch.tensor(np.array(edge_features, dtype=float)).type(torch.DoubleTensor) 
        else:
            d.edge_index_r = torch.tensor(np.array([[],[]])).type(torch.LongTensor)
            d.edge_attr_r = torch.Tensor(np.array([]))
        # predicting value
        y_array = np.array(y, dtype=float)
        if y_array.ndim == 0:
            y_array = np.array([[y_array]])
        else:
            y_array = y_array.reshape(1, -1)
        d.y = torch.tensor(y_array).type(torch.DoubleTensor)
        return d, cliques

    except Exception as e:
        print('ERROR (REDUCED FAIL):', smiles, e)
        return None, None

# generate graph fron (xi, yi, ids)
def generateGraph(d, with_id=False):
    smiles, y, ids = d
    if with_id:
        g = constructGraph(smiles, y, ids)
    else:
        g = constructGraph(smiles, y)
    return g

# generate graph datasets
def generateGraphDataset(file, datasets=None):
    path = 'dataset/'+file+'/graph.pickle'

    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_graph = pickle.load(handle)
    else:
        # read validated if exists
        path_validate = 'dataset/'+file+'/'+file+'_validated.csv'
        if os.path.exists(path_validate):
            datasets = readFromCSV(path_validate)
            datasets = datasets[datasets['v']=='valid']

        datasets_graph = dict()
        for d in tqdm(zip(datasets['X'].values, datasets['y'].values, datasets['ids'].values)):
            smiles = d[0]
            g = generateGraph(d)
            if g:
                datasets_graph[smiles] = g

        # write dataset to pickle 
        with open(path, 'wb') as handle:
            pickle.dump(datasets_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        print('Function: generateGraphDataset()')
        print('Datasets graph: ', len(datasets_graph))
    except:
        print('Cannot print dataset statistics')

    return datasets_graph

# generate graph datasets
def generateGraphDatasetUnknown(file, smiles):
    # creating file folder
    if not(os.path.isdir('./dataset/{}'.format(file))):
        os.makedirs(os.path.join('./dataset/{}'.format(file)))

    path = 'dataset/'+file+'/unk_graph.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_unk_graph = pickle.load(handle)
    else:
        path = 'dataset/'+file+'.csv'
        if os.path.exists(path):
            df = pd.read_csv('dataset/'+file+'.csv')
            X = np.array(df[smiles])
            y = np.zeros(len(df[smiles]))
            if 'Id' in df.columns:
                ids = np.array(df['Id'])
            else:
                ids = X # assign ids as original smiles
            datasets = (X, y, ids)
        else:
            print('ERROR: file does not exist.')

        datasets_unk_graph = dict()
        for d in zip(datasets[0], datasets[1], datasets[2]):
            smiles = d[0]
            g = generateGraph(d, with_id=True)
            if g:
                datasets_unk_graph[smiles] = g

        path = 'dataset/'+file+'/unk_graph.pickle'
        # write dataset to pickle 
        with open(path, 'wb') as handle:
            pickle.dump(datasets_unk_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        print('Function: generateGraphDatasetUnknown()')
        print('Datasets Test: ', len(datasets_unk_graph))
    except:
        print('Cannot print dataset statistics')

    return datasets_unk_graph

# generate vocaburary using train data of graph datasets
def generateVocabTrain(file, splitting_seed, splitting_fold=1, vocab_len=100):
    for i in range(splitting_fold):
        print('Generating vocab fold', i)
        path = 'dataset/'+file+'/train_'+str(splitting_seed)+'_'+str(i)+'.csv'
        if os.path.exists(path):
            df = readFromCSV(path)
        smiles_list = list(df['X'])
        vocab_file = file+'_'+str(i)
        vocab_path = 'vocab/'+vocab_file+'.txt'
        if os.path.exists(vocab_path):
            print('Vocab files already exist')
            return
        generate_vocab(smiles_list, vocab_len, vocab_path)

# generate tokenizer using vocab path
def generateTokenizer(vocab_file):
    vocab_path = 'vocab/'+vocab_file+'.txt'
    tokenizer = Tokenizer(vocab_path)
    return tokenizer

# generate reduced graph dict
def generateReducedGraphDict(file, reducedgraph, vocab_file=None):
    if reducedgraph == 'substructure':
        path = 'dataset/'+file+'/'+reducedgraph+'_'+vocab_file+'_final.pickle'
    else:
        path = 'dataset/'+file+'/'+reducedgraph+'_final.pickle'

    if os.path.exists(path):
        with open(path, 'rb') as handle:
            reduced_graph_dict = pickle.load(handle)
    else:
        reduced_graph_dict = {}
        duplicated = []
        tokenizer = generateTokenizer(vocab_file) if reducedgraph == 'substructure' else None

        # read validated if exists
        path_validate = 'dataset/'+file+'/'+file+'_validated.csv'
        if os.path.exists(path_validate):
            datasets = readFromCSV(path_validate)
            datasets = datasets[datasets['v']=='valid']
        X = datasets['X'].values
        y = datasets['y'].values

        for X, y in tqdm(zip(X, y)):
            smiles = X
            d, cliques = constructReducedGraph(reducedgraph, smiles, y, tokenizer)
            if d:
                if smiles in reduced_graph_dict: 
                    duplicated.append(smiles)
                reduced_graph_dict[smiles] = (d, cliques)

        print("number of duplicated smiles:", len(duplicated), duplicated)

        with open(path, 'wb') as handle:
            pickle.dump(reduced_graph_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print('Function: generateReducedGraphDict()')
    # print("number of all reduced graphs:", len(reduced_graph_dict))

    return reduced_graph_dict

# generate data loader 
def generateDataLoader(file, batch_size, seed, fold_number):
    path = 'dataset/'+file+'/graph.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_graph = pickle.load(handle)

    path_train = 'dataset/'+file+'/train_'+str(seed)+'_'+str(fold_number)+'.csv'
    datasets_train_df = readFromCSV(path_train)
    path_val = 'dataset/'+file+'/val_'+str(seed)+'_'+str(fold_number)+'.csv'
    datasets_val_df = readFromCSV(path_val)
    path_test = 'dataset/'+file+'/test.csv'
    datasets_test_df = readFromCSV(path_test)
    
    # datasets_train = [x for x in datasets_graph if x.smiles in datasets_train_df['X'].values] 
    # datasets_val = [x for x in datasets_graph if x.smiles in datasets_val_df['X'].values] 
    # datasets_test = [x for x in datasets_graph if x.smiles in datasets_test_df['X'].values] 
    datasets_train = [datasets_graph[x] for x in datasets_train_df['X'].values if x in datasets_graph] 
    datasets_val = [datasets_graph[x] for x in datasets_val_df['X'].values if x in datasets_graph] 
    datasets_test = [datasets_graph[x] for x in datasets_test_df['X'].values if x in datasets_graph] 
    loader_train = DataLoader(datasets_train, batch_size=batch_size, shuffle=True, follow_batch=['x_g'])
    loader_val = DataLoader(datasets_val, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    loader_test = DataLoader(datasets_test, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    return loader_train, loader_val, loader_test, datasets_train, datasets_val, datasets_test

# generate data loader for Testing
def generateDataLoaderTesting(file, batch_size):
    path = 'dataset/'+file+'/graph.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_graph = pickle.load(handle)

    path_test = 'dataset/'+file+'/test.csv'
    datasets_test_df = readFromCSV(path_test)

    # datasets_test = [x for x in datasets_graph if x.smiles in datasets_test_df['X'].values] 
    datasets_test = [datasets_graph[x] for x in datasets_test_df['X'].values if x in datasets_graph] 
    loader_test = DataLoader(datasets_test, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    return loader_test, datasets_test

# generate data loader for training+val
def generateDataLoaderTraining(file, batch_size):
    path = 'dataset/'+file+'/graph.pickle'
    if os.path.exists(path):
        with open(path, 'rb') as handle:
            datasets_graph = pickle.load(handle)

    path_trainval = 'dataset/'+file+'/'+file+'_validated.csv'
    datasets_trainval_df = readFromCSV(path_trainval)

    # datasets_trainval = [x for x in datasets_graph if x.smiles in datasets_trainval_df['X'].values] 
    datasets_trainval = [datasets_graph[x] for x in datasets_trainval_df['X'].values if x in datasets_graph] 
    loader_trainval = DataLoader(datasets_trainval, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    return loader_trainval, datasets_trainval

# generate data loader for list of graph data
def generateDataLoaderListing(datasets_list, batch_size):
    loader_test = DataLoader(datasets_list, batch_size=batch_size, shuffle=False, follow_batch=['x_g'])
    return loader_test, datasets_list

# get number of smiles in item
def getNumberofSmiles(item):
    if hasattr(item, 'num_graphs'):
        return item.num_graphs
    smiles = getattr(item, 'smiles', None)
    if isinstance(smiles, (list, tuple)):
        return len(smiles)
    return 1

######################
### Import Library ###
######################

# My library
from molgraph.dataset import *
from molgraph.graphmodel import *
from molgraph.training import *
from molgraph.testing import *
from molgraph.visualize import *
from molgraph.experiment import *
# General library
import os
import argparse
import numpy as np
import datetime
# pytorch
import torch
import pytorch_lightning as pl
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.set_default_dtype(torch.float64)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

#####################
### Argument List ###
#####################


def save_model_results(trainer, args, x_embed, y_test, path, model_type, use_kan):
    """保存模型训练结果"""
    try:
        # 创建结果目录
        results_dir = path
        os.makedirs(results_dir, exist_ok=True)
        
        # 收集模型信息
        model_info = {
            'model_type': model_type,
            'use_kan_readout': use_kan,
            'dataset': args.file,
            'task_type': args.graphtask,
            'experiment_name': getattr(args, 'experiment_number', 'unknown'),
            'seed': getattr(args, 'seed', 'unknown'),
            'embedding_shape': list(x_embed.shape) if hasattr(x_embed, 'shape') else 'unknown',
            'test_samples': len(y_test) if y_test is not None else 0,
            'timestamp': datetime.datetime.now().isoformat(),
            'log_folder': trainer.log_folder_name
        }
        
        
        # 合并所有结果
        results_data = {
            'model_info': model_info,
            'embeddings_info': {
                'shape': list(x_embed.shape) if hasattr(x_embed, 'shape') else None,
                'labels_shape': list(y_test.shape) if hasattr(y_test, 'shape') else None
            }
        }
        
        # 保存为JSON文件
        filename_suffix = "kan" if use_kan else "baseline"
        results_file = os.path.join(results_dir, f'training_results_{filename_suffix}.json')
        
        import json
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"✓ {model_type} 训练结果已保存: {results_file}")
        
        # 保存简单的文本信息
        info_file = os.path.join(results_dir, f'model_info_{filename_suffix}.txt')
        with open(info_file, 'w') as f:
            f.write(f"=== {model_type} 模型训练完成信息 ===\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
            f.write(f"训练完成时间: {datetime.datetime.now()}\n")
        
        print(f"✓ {model_type} 模型信息已保存: {info_file}")
        
        return results_file
        
    except Exception as e:
        print(f"保存 {model_type} 模型结果时出错: {e}")
        return None




####################
### Main Program ###
####################

if __name__ == '__main__':
    print(os.environ["CUBLAS_WORKSPACE_CONFIG"])
    parser = ArgumentParser()
    args = parser.getArgument()
    print(args)

    file = args.file
    smiles = args.smiles 
    task = args.task
    splitting = args.splitting 
    splitting_fold = args.fold
    splitting_seed = args.splitting_seed

    # 检测是否使用KAN
    use_kan = getattr(args, 'use_kan_readout', False)
    model_type = "KAN" if use_kan else "Baseline"
    
    print(f"{'='*50}")
    print(f"训练模式: {model_type}")
    print(f"使用KAN Readout: {use_kan}")
    print(f"{'='*50}")

    # get validated dataset
    datasets = getDataset(file, smiles, task, splitting)
    # compute positive weight for classification
    if args.graphtask == 'classification':
        args.pos_weight = getPosWeight(datasets)
        print('pos_weight:', args.pos_weight)
    # generate dataset splitting
    datasets_splitted = generateDatasetSplitting(file, splitting, splitting_fold, splitting_seed)
    # generate all graph dataset
    datasets_graph = generateGraphDataset(file)
    # generate all reduced graph dataset
    dict_reducedgraph = dict()
    for g in args.reduced:
        if g == 'substructure':
            for i in range(splitting_fold):
                vocab_file = file+'_'+str(i)
                if not os.path.exists('vocab/'+vocab_file+'.txt'):
                    generateVocabTrain(file, splitting_seed, splitting_fold, vocab_len=args.vocab_len)
                dict_reducedgraph[g] = generateReducedGraphDict(file, g, vocab_file=vocab_file)
        else:
            dict_reducedgraph[g] = generateReducedGraphDict(file, g)
        
    trainer = Trainer(args)
    trainer.train()

    args_test = dict()
    # Load model
    args_test['log_folder_name'] = trainer.log_folder_name
    args_test['exp_name'] = args.experiment_number
    args_test['fold_number'] = 0
    args_test['seed'] = args.seed

    test_loader, datasets_test =  generateDataLoaderTesting(args.file, args.batch_size)

    tester = Tester(args, args_test)
    tester.test(test_loader)

    x_embed = tester.getXEmbed()
    y_test = tester.getYTest()
    path = 'dataset/'+trainer.log_folder_name+'/results'
    legend = getLegend(args.graphtask, y_test)

    # 原有的可视化
    visualize_pca(x_embed, y_test, title=args.file, path=path, legend=legend)
    visaulize_tsne(x_embed, y_test, title=args.file, path=path, legend=legend)

    
    # 保存训练结果用于后续对比
    print(f"保存 {model_type} 模型训练信息...")
    save_model_results(trainer, args, x_embed, y_test, path, model_type, use_kan)

    print('COMPLETED!')

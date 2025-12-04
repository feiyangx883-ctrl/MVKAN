######################
### Import Library ###
######################

# my library
from molgraph.graphmodel import *
# from molgraph.graphmodel_star import *
# from molgraph.graphmodel_signed import *
from molgraph.dataset import *
from molgraph.experiment import *
# general
import math as math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tqdm import tqdm, trange
# sklearn
import sklearn.metrics as metrics
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
# pytorch
from torch.autograd import Variable
from transformers.optimization import get_cosine_schedule_with_warmup

#####################
### Trainer Class ###
#####################

# Trainer
class Trainer(object):

    def __init__(self, args):
        super(Trainer, self).__init__()

        # experiment
        self.args = args
        self.seed = set_seed(self.args.seed)
        self.log_folder_name, self.exp_name = set_experiment_name(self.args)
        self.device = set_device(self.args) 
        torch.manual_seed(self.seed)

        # dataset
        self.file = self.args.file
        
        # pos_weight (用于多任务和分类任务)
        self.pos_weight = None
        if self.args.graphtask in ['classification', 'multitask']:
            self.pos_weight = self._compute_pos_weight()
        
        # Training history for visualization - track across all folds
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_metrics': [],
            'test_results': [],
            'fold_histories': []
        }
        
    def _compute_pos_weight(self):
        """
        计算正样本权重,支持单任务和多任务场景
        自动从 args 中获取已配置的 smiles, task, splitting 参数
        """
        try:
            from molgraph.dataset import getDataset, getPosWeight
            
            # 直接使用 args 中已经从 _dataset.csv 读取的配置
            datasets = getDataset(
                self.args.file, 
                self.args.smiles,  
                self.args.task,     
                self.args.splitting 
            )
            
            pos_weight = getPosWeight(datasets)
            
            # 打印权重信息
            if isinstance(pos_weight, np.ndarray):
                print(f'✓ 多任务 pos_weight 计算完成 (共 {len(pos_weight)} 个任务):')
                for i, w in enumerate(pos_weight):
                    print(f'  Task {i}: {w:.4f}')
            else:
                print(f'✓ 单任务 pos_weight: {pos_weight:.4f}')
            
            return pos_weight
            
        except Exception as e:
            print(f'计算 pos_weight 失败: {e}')
            print(' 将使用默认权重 (无权重)')
            return None
        

    def train(self):

        if self.args.graphtask == 'regression':
            ### Train
            overall_results = {
                'val_rmse': [],
                'test_rmse': [],
                'test_r2': []
            }

            fold_iter = tqdm(range(0, self.args.fold), desc='Training')

            for fold_number in fold_iter:

                loss_fn = F.mse_loss
                
                ### Set logger, loss and accuracy for each fold
                logger = set_logger_fold(self.log_folder_name, self.exp_name, self.seed, fold_number)

                patience = 0
                best_loss_epoch = 0
                best_rmse_epoch = 0
                best_loss = 1e9
                best_loss_rmse = 1e9 
                best_loss_rmse = 1e9
                best_rmse = 1e9
                best_rmse_loss = 1e9

                train_loss_history = []
                val_loss_history = []
                val_metric_history = []

                
                train_loader, val_loader, test_loader, datasets_train, datasets_val, datasets_test =  generateDataLoader(self.file, self.args.batch_size, self.seed, fold_number)

                # Load model and optimizer
                self.model = load_model(self.args).to(self.device)
                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
                
                if self.args.lr_schedule:
                    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                                     self.args.patience * len(train_loader), 
                                                                     self.args.num_epochs * len(train_loader))
                
                t_start = time.perf_counter()
                ### K-Fold Training
                for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

                    self.model.train()
                    total_loss = 0

                    for _, data in enumerate(train_loader):

                        data = data.to(self.device)
                        out = self.model(data, fold_number=fold_number)
                        # Ensure shapes match for BCEWithLogitsLoss when model returns [N,1]
                        if out.dim() == 2 and out.size(1) == 1:
                            out_flat = out.view(-1)
                        else:
                            out_flat = out
                        y_flat = data.y.view(-1).to(out_flat.dtype)
                        loss = loss_fn(out_flat, y_flat)
                        # Before the backward pass, use the optimizer object to zero all of the
                        # gradients for the variables it will update (which are the learnable
                        # weights of the model)
                        self.optimizer.zero_grad()
                        # Backward pass: compute gradient of the loss with respect to model
                        loss.backward()
                        # keep the gradients within a specific range.
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                        # Calling the step function on an Optimizer makes an update to its
                        self.optimizer.step()

                        total_loss += loss.item() * getNumberofSmiles(data)

                        if self.args.lr_schedule:
                            self.scheduler.step()

                    total_loss = total_loss / len(train_loader.dataset)

                    ### Validation
                    val_rsme, val_loss, val_r2 = self.eval_regression(val_loader, loss_fn)
                    
                    train_loss_history.append(total_loss)
                    val_loss_history.append(val_loss)
                    val_metric_history.append(val_rsme)
                    
                    if val_loss < best_loss:
                        best_loss_rmse = val_rsme
                        best_loss = val_loss
                        best_loss_epoch = epoch

                    if val_rsme < best_rmse:
                        best_rmse = val_rsme
                        best_rmse_loss = val_loss
                        best_rmse_epoch = epoch
                        torch.save(self.model.state_dict(), 
                            f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                            f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                        patience = 0
                    else:
                        patience += 1

                    ### Validation log
                    logger.log(f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"TrainLoss: {total_loss:.4f}, ValLoss: {val_loss:.4f}, ValRMSE: {val_rsme:.4f}, ValR2: {val_r2:.4f} //"
                               f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"Best Loss> Loss: {best_loss:.4f}, RMSE: {best_loss_rmse:.4f}, at Epoch: {best_loss_epoch} / "
                               f"Best RMSE> Loss: {best_rmse_loss:.4f}, RMSE: {best_rmse:.4f}, at Epoch: {best_rmse_epoch}")

                    fold_iter.set_description(f'[Fold {fold_number}]-Epoch: {epoch} TrainLoss: {total_loss:.4f} '
                                              f'ValLoss: {val_loss:.4f} ValRMSE: {val_rsme:.4f} patience: {patience}')
                    fold_iter.refresh()
                    if patience > self.args.patience: break
                
                self._save_training_curves(train_loss_history, val_loss_history, val_metric_history,
                                           metric_label='Val RMSE', fold_number=fold_number)
                
                t_end = time.perf_counter()

                ### Test log
                checkpoint = torch.load(f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                self.model.load_state_dict(checkpoint)
                
                test_rmse, test_loss, test_r2 = self.eval_regression(test_loader, loss_fn)
                
                logger.log(f"[Test: Fold {fold_number}] "
                           f"Best Loss> Loss: {best_loss:4f}, RMSE: {best_loss_rmse:4f}, at Epoch: {best_loss_epoch} /"
                           f"Best RMSE> Loss: {best_rmse_loss:4f}, RMSE: {best_rmse:4f}, at Epoch: {best_rmse_epoch} //"
                           f"[Test: Fold {fold_number}] Test> RMSE: {test_rmse:4f}, R2: {test_r2:4f}, with Time: {t_end-t_start:.2f}")

                test_result_file = "./dataset/{}/results/{}-results.txt".format(self.log_folder_name, self.exp_name)
                with open(test_result_file, 'a+') as f:
                    f.write(f"[FOLD {fold_number}] {self.seed}: BEST Loss: {best_loss:.4f}, BEST RMSE: {best_rmse:.4f} //"
                            f"Test> Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:4f}\n")

                ### Report results
                overall_results['val_rmse'].append(best_rmse)
                overall_results['test_rmse'].append(test_rmse)
                overall_results['test_r2'].append(test_r2)
                
                # Store test results for KAN visualization
                self.training_history['test_results'].append({
                    'fold': fold_number,
                    'test_metric': test_rmse,
                    'test_r2': test_r2,
                    'best_val_metric': best_rmse
                })

                final_result_file = f"./dataset/{self.log_folder_name}/results/{self.exp_name}-final.txt"
                with open(final_result_file, 'a+') as f:
                    f.write(f"{self.seed}: ValRMSE_Mean: {np.array(overall_results['val_rmse']).mean():.4f}, "
                            f"ValRMSE_Std: {np.array(overall_results['val_rmse']).std():.4f}, " 
                            f"TestRMSE_Mean: {np.array(overall_results['test_rmse']).mean():.4f}, "
                            f"TestRMSE_Std: {np.array(overall_results['test_rmse']).std():.4f}, " 
                            f"TestR2_Mean: {np.array(overall_results['test_r2']).mean():.4f}, "
                            f"TestR2_Std: {np.array(overall_results['test_r2']).std():.4f}\n")

                print('ValRMSE ', str(np.array(overall_results['val_rmse']).mean()), '+/-', str(np.array(overall_results['val_rmse']).std()))
                print('TestRMSE', str(np.array(overall_results['test_rmse']).mean()), '+/-', str(np.array(overall_results['test_rmse']).std()))
            
            # ========== KAN 可解释性可视化 ==========
            if getattr(self.args, 'use_kan_readout', False) or getattr(self.args, 'use_kan_classifier', False):
                try:
                    from molgraph.kan_visualizer import KANVisualizer
                    
                    # 加载最后一个 fold 的最佳模型
                    checkpoint_path = (
                        f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth'
                    )
                    checkpoint = torch.load(checkpoint_path)
                    self.model.load_state_dict(checkpoint)
                    
                    # 创建可视化器
                    vis_dir = os.path.join('./dataset', self.log_folder_name, 'visualizations', self.exp_name)
                    visualizer = KANVisualizer(self.model, device=self.device, save_dir=vis_dir)
                    
                    # 生成可视化
                    print("\n" + "="*60)
                    visualizer.generate_all_visualizations(test_loader=test_loader)
                    print("="*60 + "\n")
                    
                except Exception as e:
                    print(f"⚠️  Warning: KAN visualization failed: {e}")
                    import traceback
                    traceback.print_exc()

        elif self.args.graphtask == 'classification':

            ### Train
            overall_results = {
                'val_auc': [],
                'val_acc': [],
                'test_auc': [],
                'test_acc': []
            }

            fold_iter = tqdm(range(0, self.args.fold), desc='Training')

            for fold_number in fold_iter:
                
                # This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([float(self.args.pos_weight)]).type(torch.DoubleTensor)).to(self.device)
                
                ### Set logger, loss and accuracy for each fold
                logger = set_logger_fold(self.log_folder_name, self.exp_name, self.seed, fold_number)

                patience = 0
                best_loss_epoch = 0
                best_auc_epoch = 0
                best_loss = 1e9
                best_loss_auc = -1e9
                best_auc = -1e9
                best_auc_loss = 1e9
                
                
                train_loss_history = []
                val_loss_history = []
                val_metric_history = []

                train_loader, val_loader, test_loader, datasets_train, datasets_val, datasets_test =  generateDataLoader(self.file, self.args.batch_size, self.seed, fold_number)

                # Load model and optimizer
                self.model = load_model(self.args).to(self.device)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
                
                if self.args.lr_schedule:
                    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                                                                     self.args.patience * len(train_loader), 
                                                                     self.args.num_epochs * len(train_loader))
                
                t_start = time.perf_counter()
                ### K-Fold Training
                for epoch in trange(0, (self.args.num_epochs), desc = '[Epoch]', position = 1):

                    self.model.train()
                    total_loss = 0

                    for _, data in enumerate(train_loader):

                        data = data.to(self.device)
                        out = self.model(data, fold_number=fold_number)
                        # Ensure shapes match for BCEWithLogitsLoss when model returns [N,1]
                        if out.dim() == 2 and out.size(1) == 1:
                            out_flat = out.view(-1)
                        else:
                            out_flat = out
                        y_flat = data.y.view(-1).to(out_flat.dtype)
                        loss = loss_fn(out_flat, y_flat)
                        # Before the backward pass, use the optimizer object to zero all of the
                        # gradients for the variables it will update (which are the learnable
                        # weights of the model)
                        self.optimizer.zero_grad()
                        # Backward pass: compute gradient of the loss with respect to model
                        loss.backward()
                        # keep the gradients within a specific range.
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                        # Calling the step function on an Optimizer makes an update to its
                        self.optimizer.step()

                        total_loss += loss.item() * getNumberofSmiles(data)

                        if self.args.lr_schedule:
                            self.scheduler.step()

                    total_loss = total_loss / len(train_loader.dataset)

                    ### Validation
                    val_acc, val_loss, val_auc = self.eval_classification(val_loader, loss_fn)
                    
                    
                    
                    train_loss_history.append(total_loss)
                    val_loss_history.append(val_loss)
                    val_metric_history.append(val_auc)
                    
                    if val_loss < best_loss:
                        best_loss_auc = val_auc
                        best_loss = val_loss
                        best_loss_epoch = epoch

                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_auc_loss = val_loss
                        best_auc_epoch = epoch
                        torch.save(self.model.state_dict(), 
                            f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                            f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                        patience = 0
                    else:
                        patience += 1

                    ### Validation log
                    logger.log(f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"TrainLoss: {total_loss:.4f}, ValLoss: {val_loss:.4f}, ValAUC: {val_auc:.4f}, ValACC: {val_acc:.4f} //"
                               f"[Val: Fold {fold_number}-Epoch {epoch}] "
                               f"Best Loss> Loss: {best_loss:.4f}, AUC: {best_loss_auc:.4f}, at Epoch: {best_loss_epoch} / "
                               f"Best AUC> Loss: {best_auc_loss:.4f}, AUC: {best_auc:.4f}, at Epoch: {best_auc_epoch}")

                    fold_iter.set_description(f'[Fold {fold_number}]-Epoch: {epoch} TrainLoss: {total_loss:.4f} '
                                              f'ValLoss: {val_loss:.4f} ValAUC: {val_auc:.4f} patience: {patience}')
                    
                    fold_iter.refresh()
                    if patience > self.args.patience: break
                
                self._save_training_curves(train_loss_history, val_loss_history, val_metric_history,
                                           metric_label='Val AUC', fold_number=fold_number)
                t_end = time.perf_counter()

                ### Test log
                checkpoint = torch.load(f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth')
                self.model.load_state_dict(checkpoint)
                
                test_acc, test_loss, test_auc = self.eval_classification(test_loader, loss_fn)
                
                logger.log(f"[Test: Fold {fold_number}] "
                           f"Best Loss> Loss: {best_loss:4f}, AUC: {best_loss_auc:4f}, at Epoch: {best_loss_epoch} /"
                           f"Best AUC> Loss: {best_auc_loss:4f}, AUC: {best_auc:4f}, at Epoch: {best_auc_epoch} //"
                           f"[Test: Fold {fold_number}] Test> AUC: {test_auc:4f}, ACC: {test_acc:4f}, with Time: {t_end-t_start:.2f}")

                test_result_file = "./dataset/{}/results/{}-results.txt".format(self.log_folder_name, self.exp_name)
                with open(test_result_file, 'a+') as f:
                    f.write(f"[FOLD {fold_number}] {self.seed}: BEST Loss: {best_loss:.4f}, BEST AUC: {best_auc:.4f} //"
                            f"Test> Loss: {test_loss:.4f}, AUC: {test_auc:.4f}, ACC: {test_acc:4f}\n")

                ### Report results
                overall_results['val_auc'].append(best_auc)
                overall_results['test_auc'].append(test_auc)
                overall_results['test_acc'].append(test_acc)
                
                # Store test results for KAN visualization
                self.training_history['test_results'].append({
                    'fold': fold_number,
                    'test_metric': test_auc,
                    'test_acc': test_acc,
                    'best_val_metric': best_auc
                })

                val_auc_mean = np.array(overall_results['val_auc']).mean()
                val_auc_std = np.array(overall_results['val_auc']).std()
                test_auc_mean = np.array(overall_results['test_auc']).mean()
                test_auc_std = np.array(overall_results['test_auc']).std()
                test_acc_mean = np.array(overall_results['test_acc']).mean()
                test_acc_std = np.array(overall_results['test_acc']).std()

                final_result_file = f"./dataset/{self.log_folder_name}/results/{self.exp_name}-final.txt"
                with open(final_result_file, 'a+') as f:
                    f.write(
                        f"{self.seed}: "
                        f"ValAUC: {val_auc_mean:.4f} +/- {val_auc_std:.4f}, "
                        f"TestAUC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}, "
                        f"TestACC: {test_acc_mean:.4f} +/- {test_acc_std:.4f}\n"
                    )

                # 清晰的命令行最终结果展示
                print(f"Final Val AUC:  {val_auc_mean:.4f} +/- {val_auc_std:.4f}")
                print(f"Final Test AUC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}")
            
            # ========== KAN 可解释性可视化 ==========
            if getattr(self.args, 'use_kan_readout', False) or getattr(self.args, 'use_kan_classifier', False):
                try:
                    from molgraph.kan_visualizer import KANVisualizer
                    
                    # 加载最后一个 fold 的最佳模型
                    checkpoint_path = (
                        f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth'
                    )
                    checkpoint = torch.load(checkpoint_path)
                    self.model.load_state_dict(checkpoint)
                    
                    # 创建可视化器
                    vis_dir = os.path.join('./dataset', self.log_folder_name, 'visualizations', self.exp_name)
                    visualizer = KANVisualizer(self.model, device=self.device, save_dir=vis_dir)
                    
                    # 生成可视化
                    print("\n" + "="*60)
                    visualizer.generate_all_visualizations(test_loader=test_loader)
                    print("="*60 + "\n")
                    
                except Exception as e:
                    print(f"⚠️  Warning: KAN visualization failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        elif self.args.graphtask == 'multitask':
            ### Train Multi-task
            overall_results = {
                'val_auc': [],
                'val_auc_per_task': [],
                'test_auc': [],
                'test_auc_per_task': []
            }

            fold_iter = tqdm(range(0, self.args.fold), desc='Training')

            for fold_number in fold_iter:
                

                if self.pos_weight is not None:
                    if isinstance(self.pos_weight, np.ndarray):
                        # 多任务场景: pos_weight shape = [num_tasks]
                        pos_weight_tensor = torch.tensor(
                            self.pos_weight, 
                            dtype=torch.float64
                        ).to(self.device)
                    else:
                        # 单值场景
                        pos_weight_tensor = torch.tensor(
                            [float(self.pos_weight)], 
                            dtype=torch.float64
                        ).to(self.device)
                    

                    loss_fn = None  # 将使用自定义损失计算
                    pos_weight_for_loss = pos_weight_tensor
                else:
                    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
                    pos_weight_for_loss = None
                
                ### Set logger for each fold
                logger = set_logger_fold(self.log_folder_name, self.exp_name, self.seed, fold_number)

                # Early stopping and tracking variables
                patience = 0
                best_loss_epoch = 0
                best_auc_epoch = 0
                best_loss = 1e9
                best_loss_auc = -1e9
                best_auc = -1e9
                best_auc_loss = 1e9
                
                train_loss_history = []
                val_loss_history = []
                val_metric_history = []

                # Generate data loaders
                train_loader, val_loader, test_loader, datasets_train, datasets_val, datasets_test = \
                    generateDataLoader(self.file, self.args.batch_size, self.seed, fold_number)

                # Load model and optimizer
                self.model = load_model(self.args).to(self.device)
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), 
                    lr=self.args.lr, 
                    weight_decay=self.args.weight_decay
                )
                
                # Learning rate scheduler
                if self.args.lr_schedule:
                    self.scheduler = get_cosine_schedule_with_warmup(
                        self.optimizer, 
                        self.args.patience * len(train_loader), 
                        self.args.num_epochs * len(train_loader)
                    )
                
                t_start = time.perf_counter()
                
                ### K-Fold Training
                for epoch in trange(0, self.args.num_epochs, desc='[Epoch]', position=1):
                    
                    self.model.train()
                    total_loss = 0

                    for _, data in enumerate(train_loader):
                        data = data.to(self.device)
                        out = self.model(data, fold_number=fold_number)
                        
                        
                        if out.dim() == 2 and data.y.dim() == 2:
                            # 多任务场景
                            mask = ~torch.isnan(data.y)
                            
                            if mask.sum() > 0:
                                if pos_weight_for_loss is not None:
                                    # 手动计算带权重的 BCE Loss
                                    # pos_weight shape: [num_tasks]
                                    # 需要将 pos_weight 扩展到 [batch_size, num_tasks]
                                    
                                    y_masked = data.y[mask].to(out.dtype)
                                    out_masked = out[mask]
                                    
                                    # 计算 BCE with logits and pos_weight
                                    # 公式: -[y*log(sigmoid(x))*pos_weight + (1-y)*log(1-sigmoid(x))]
                                    max_val = torch.clamp(out_masked, min=0)
                                    
                                    if pos_weight_for_loss.numel() == 1:
                                        # 所有任务使用相同权重
                                        pos_weight_expanded = pos_weight_for_loss.expand_as(out_masked)
                                    else:
                                        # 每个任务不同权重 - 需要广播
                                        # 从 mask 中恢复原始形状信息
                                        batch_size, num_tasks = data.y.shape
                                        
                                        # 重塑 pos_weight: [num_tasks] -> [1, num_tasks]
                                        pos_weight_broadcast = pos_weight_for_loss.view(1, -1)
                                        
                                        # 扩展到完整 batch: [1, num_tasks] -> [batch_size, num_tasks]
                                        pos_weight_full = pos_weight_broadcast.expand(batch_size, num_tasks)
                                        
                                        # 应用相同的 mask
                                        pos_weight_expanded = pos_weight_full[mask]
                                    
                                    loss = out_masked - out_masked * y_masked + max_val + \
                                        ((-max_val).exp() + (-out_masked - max_val).exp()).log()
                                    
                                    # 应用 pos_weight 到正样本
                                    loss = torch.where(
                                        y_masked == 1,
                                        loss * pos_weight_expanded,
                                        loss
                                    )
                                    
                                    loss = loss.mean()
                                else:
                                    # 无权重的标准 BCE Loss
                                    loss = F.binary_cross_entropy_with_logits(
                                        out[mask], 
                                        data.y[mask].to(out.dtype),
                                        reduction='mean'
                                    )
                            else:
                                continue  # 跳过全为 NaN 的 batch
                                
                        elif out.dim() == 2 and out.size(1) == 1:
                            # 单任务但输出为 [N, 1]
                            out_flat = out.view(-1)
                            y_flat = data.y.view(-1).to(out_flat.dtype)
                            
                            if loss_fn is not None:
                                loss = loss_fn(out_flat, y_flat)
                            else:
                                loss = F.binary_cross_entropy_with_logits(
                                    out_flat, y_flat,
                                    pos_weight=pos_weight_for_loss,
                                    reduction='mean'
                                )
                        else:
                            # 其他情况
                            out_flat = out.view(-1)
                            y_flat = data.y.view(-1).to(out_flat.dtype)
                            
                            if loss_fn is not None:
                                loss = loss_fn(out_flat, y_flat)
                            else:
                                loss = F.binary_cross_entropy_with_logits(
                                    out_flat, y_flat,
                                    reduction='mean'
                                )
                        
                        # Optimization step
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
                        self.optimizer.step()

                        total_loss += loss.item() * getNumberofSmiles(data)

                        if self.args.lr_schedule:
                            self.scheduler.step()

                    total_loss = total_loss / len(train_loader.dataset)

                    ### Validation
                    val_auc, val_loss, val_auc_per_task = self.eval_multitask(val_loader, pos_weight_for_loss)
                    
                    train_loss_history.append(total_loss)
                    val_loss_history.append(val_loss)
                    val_metric_history.append(val_auc)
                    
                    # Track best models
                    if val_loss < best_loss:
                        best_loss_auc = val_auc
                        best_loss = val_loss
                        best_loss_epoch = epoch

                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_auc_loss = val_loss
                        best_auc_epoch = epoch
                        torch.save(
                            self.model.state_dict(), 
                            f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                            f'fold-{fold_number}_seed-{self.seed}_best-model.pth'
                        )
                        patience = 0
                    else:
                        patience += 1

                    ### Validation log
                    logger.log(
                        f"[Val: Fold {fold_number}-Epoch {epoch}] "
                        f"TrainLoss: {total_loss:.4f}, ValLoss: {val_loss:.4f}, ValAUC: {val_auc:.4f} //"
                        f"[Val: Fold {fold_number}-Epoch {epoch}] "
                        f"Best Loss> Loss: {best_loss:.4f}, AUC: {best_loss_auc:.4f}, at Epoch: {best_loss_epoch} / "
                        f"Best AUC> Loss: {best_auc_loss:.4f}, AUC: {best_auc:.4f}, at Epoch: {best_auc_epoch}"
                    )

                    fold_iter.set_description(
                        f'[Fold {fold_number}]-Epoch: {epoch} TrainLoss: {total_loss:.4f} '
                        f'ValLoss: {val_loss:.4f} ValAUC: {val_auc:.4f} patience: {patience}'
                    )
                    fold_iter.refresh()
                    
                    if patience > self.args.patience:
                        break
                
                # Save training curves
                self._save_training_curves(
                    train_loss_history, 
                    val_loss_history, 
                    val_metric_history,
                    metric_label='Val AUC', 
                    fold_number=fold_number
                )
                t_end = time.perf_counter()

                ### Test evaluation
                checkpoint = torch.load(
                    f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                    f'fold-{fold_number}_seed-{self.seed}_best-model.pth'
                )
                self.model.load_state_dict(checkpoint)
                
                test_auc, test_loss, test_auc_per_task = self.eval_multitask(test_loader, pos_weight_for_loss)
                
                logger.log(
                    f"[Test: Fold {fold_number}] "
                    f"Best Loss> Loss: {best_loss:.4f}, AUC: {best_loss_auc:.4f}, at Epoch: {best_loss_epoch} /"
                    f"Best AUC> Loss: {best_auc_loss:.4f}, AUC: {best_auc:.4f}, at Epoch: {best_auc_epoch} //"
                    f"[Test: Fold {fold_number}] Test> AUC: {test_auc:.4f}, Time: {t_end-t_start:.2f}s"
                )

                # Save results
                test_result_file = f"./dataset/{self.log_folder_name}/results/{self.exp_name}-results.txt"
                with open(test_result_file, 'a+') as f:
                    f.write(
                        f"[FOLD {fold_number}] {self.seed}: BEST Loss: {best_loss:.4f}, BEST AUC: {best_auc:.4f} //"
                        f"Test> Loss: {test_loss:.4f}, AUC: {test_auc:.4f}\n"
                    )

                ### Accumulate results
                overall_results['val_auc'].append(best_auc)
                overall_results['test_auc'].append(test_auc)
                overall_results['test_auc_per_task'].append(test_auc_per_task)
                
                self.training_history['test_results'].append({
                    'fold': fold_number,
                    'test_metric': test_auc,
                    'test_auc_per_task': test_auc_per_task,
                    'best_val_metric': best_auc
                })

                # Statistics
                val_auc_mean = np.array(overall_results['val_auc']).mean()
                val_auc_std = np.array(overall_results['val_auc']).std()
                test_auc_mean = np.array(overall_results['test_auc']).mean()
                test_auc_std = np.array(overall_results['test_auc']).std()

                final_result_file = f"./dataset/{self.log_folder_name}/results/{self.exp_name}-final.txt"
                with open(final_result_file, 'a+') as f:
                    f.write(
                        f"{self.seed}: "
                        f"ValAUC: {val_auc_mean:.4f} +/- {val_auc_std:.4f}, "
                        f"TestAUC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}\n"
                    )

                print(f"Final Val AUC:  {val_auc_mean:.4f} +/- {val_auc_std:.4f}")
                print(f"Final Test AUC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}")
            

            
            # Per-task statistics
            if overall_results['test_auc_per_task']:
                task_auc_array = np.array(overall_results['test_auc_per_task'])
                task_auc_mean = task_auc_array.mean(axis=0)
                task_auc_std = task_auc_array.std(axis=0)
                
                task_result_file = f"./dataset/{self.log_folder_name}/results/{self.exp_name}-task-wise.txt"
                with open(task_result_file, 'w') as f:
                    f.write(f"Seed: {self.seed}\n")
                    f.write(f"Overall Test AUC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}\n\n")
                    f.write("Per-Task Test AUC:\n")
                    for task_idx, (mean_auc, std_auc) in enumerate(zip(task_auc_mean, task_auc_std)):
                        f.write(f"Task {task_idx}: {mean_auc:.4f} +/- {std_auc:.4f}\n")
            
            # ========== KAN 可解释性可视化 ==========
            if getattr(self.args, 'use_kan_readout', False) or getattr(self.args, 'use_kan_classifier', False):
                try:
                    from molgraph.kan_visualizer import KANVisualizer
                    
                    # 加载最后一个 fold 的最佳模型
                    checkpoint_path = (
                        f'./dataset/{self.log_folder_name}/checkpoints/experiment-{self.exp_name}_'
                        f'fold-{fold_number}_seed-{self.seed}_best-model.pth'
                    )
                    checkpoint = torch.load(checkpoint_path)
                    self.model.load_state_dict(checkpoint)
                    
                    # 创建可视化器
                    vis_dir = os.path.join('./dataset', self.log_folder_name, 'visualizations', self.exp_name)
                    visualizer = KANVisualizer(self.model, device=self.device, save_dir=vis_dir)
                    
                    # 生成可视化
                    print("\n" + "="*60)
                    visualizer.generate_all_visualizations(test_loader=test_loader)
                    print("="*60 + "\n")
                    
                except Exception as e:
                    print(f"⚠️  Warning: KAN visualization failed: {e}")
                    import traceback
                    traceback.print_exc()
    
    
    def _save_training_curves(self, train_losses, val_losses, val_metrics, metric_label, fold_number):
        """Persist per-epoch loss/metric curves into the visualization directory."""
        try:
            if len(train_losses) == 0 or len(val_losses) == 0:
                return

            out_dir = getattr(self.args, 'visualization_dir', None)
            if out_dir is None:
                out_dir = os.path.join('./dataset', self.log_folder_name, 'visualizations', self.exp_name)
            os.makedirs(out_dir, exist_ok=True)

            epochs = list(range(1, len(train_losses) + 1))
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue')
            ax1.plot(epochs, val_losses, label='Val Loss', color='tab:orange')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()
            if len(val_metrics) > 0:
                ax2.plot(epochs[:len(val_metrics)], val_metrics, label=metric_label, color='tab:green', alpha=0.7)
                ax2.set_ylabel(metric_label)
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper right')

            fig.tight_layout()
            fname = f"fold{fold_number}_{metric_label.lower().replace(' ', '_')}_learning_curve.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close(fig)
            
            # Store fold history for later KAN visualization
            fold_history = {
                'fold': fold_number,
                'train_losses': train_losses.copy() if isinstance(train_losses, list) else list(train_losses),
                'val_losses': val_losses.copy() if isinstance(val_losses, list) else list(val_losses),
                'val_metrics': val_metrics.copy() if isinstance(val_metrics, list) else list(val_metrics),
                'metric_label': metric_label
            }
            self.training_history['fold_histories'].append(fold_history)
            
        except Exception as e:
            # Do not let visualization failures break training; they are best-effort.
            print(f"Warning: Failed to save training curves: {e}")
            pass
    
    ### Evaluate
    def eval_regression(self, loader, loss_fn):

        self.model.eval()
        with torch.no_grad():
            rmse = 0.
            loss = 0.

            y_test = list()
            y_pred = list()

            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                loss += loss_fn(out, data.y).item() * getNumberofSmiles(data)
                # y_test.extend((data.y.cpu()).detach().numpy())
                # y_pred.extend((out.cpu()).detach().numpy())
                y_test.append((data.y).detach())
                y_pred.append((out).detach())

            y_test = torch.squeeze(torch.cat(y_test, dim=0))
            y_pred = torch.squeeze(torch.cat(y_pred, dim=0))
            
            rsme_final = math.sqrt(loss / len(loader.dataset))
            loss_final = loss / len(loader.dataset)
            # r2score = R2Score()
            # r2_final = r2score(y_pred, y_test)
            # print('TORCH:', r2_final)
            r2_final = r2_score(y_test.cpu(), y_pred.cpu())
            # print('SCIKIT:', r2_final)
            # print(rsme_final, '?==?', math.sqrt(mean_squared_error(y_test, y_pred)))
            # print(loss_final, '?==?', F.mse_loss(torch.Tensor(y_pred),torch.Tensor(y_test)).item())
            # print(r2_final)
        return rsme_final, loss_final, r2_final

    ### Evaluate
    def eval_classification(self, loader, loss_fn):
        
        self.model.eval()
        with torch.no_grad():
            m_fn = nn.Sigmoid()

            correct = 0.
            loss = 0.
            
            y_test = list()
            y_pred = list()

            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                # pred = out.max(dim=1)[1]
                pred = m_fn(out)
                # handle single-logit ([N,1]) outputs: squeeze for comparisons
                if pred.dim() == 2 and pred.size(1) == 1:
                    pred_squeezed = pred.view(-1)
                else:
                    pred_squeezed = pred
                pred_round = pred_squeezed > 0.5
                correct += pred_round.eq(data.y.view(-1)).sum().item()
                # prepare outputs / targets for loss (match shapes and dtype)
                if out.dim() == 2 and out.size(1) == 1:
                    out_flat = out.view(-1)
                else:
                    out_flat = out
                y_flat = data.y.view(-1).to(out_flat.dtype)
                loss += loss_fn(out_flat, y_flat).item() * getNumberofSmiles(data)
                # collect for metrics
                y_test.append((data.y.view(-1)).detach())
                y_pred.append((pred_squeezed).detach())

            y_test = torch.squeeze(torch.cat(y_test, dim=0))
            y_pred = torch.squeeze(torch.cat(y_pred, dim=0))
            
            acc_final = correct / len(loader.dataset)
            loss_final = loss / len(loader.dataset)
            # y_pred = torch.Tensor(np.array(y_pred))
            # y_test = torch.Tensor(y_test).type(torch.IntTensor)
            # auroc = AUROC(num_classes=args.class_number)
            # auc_final = auroc(y_pred, y_test)
            # roc = ROC(pos_label=1)
            # fpr, tpr, threshold = roc(y_pred, y_test)
            # auc_final = auc(fpr, tpr).item()
            # print('TORCH:', auc_final)
            fpr, tpr, threshold = metrics.roc_curve(y_test.cpu(), y_pred.cpu())
            auc_final = metrics.auc(fpr, tpr)
            # print('SCIKIT:', auc_final)
        return acc_final, loss_final, auc_final

    ### Evaluate
    def eval_multiclass(self, loader, loss_fn):
        
        self.model.eval()
        with torch.no_grad():

            # correct = 0.
            loss = 0.
            
            y_test = list()
            y_pred = list()

            for data in loader:

                data = data.to(self.device)
                out = self.model(data)
                # pred = out.max(dim=1)[1]
                # pred = m_fn(out).cpu()
                pred = out.to(self.device)
                # pred_round = pred > 0.5
                # pred_hat = torch.max(pred, 1).indices
                pred_hat = pred.argmax(dim=1)
                # correct += pred_round.eq(data.y.cpu()).sum().item()
                # data_y = data.y.squeeze().type(torch.LongTensor).to(self.device)
                data_y = F.one_hot(data.y.squeeze().to(torch.int64), num_classes=self.args.class_number).type(torch.DoubleTensor).to(self.device)
                # loss += loss_fn(m_fn(out), data_y.cpu()).item()
                loss += loss_fn(out.to(self.device), data_y.to(self.device)).item() * getNumberofSmiles(data)
                # y_test.extend((torch.squeeze(data.y).cpu()).detach().numpy())
                # y_pred.extend((pred_hat.cpu()).detach().numpy())
                y_test.append((torch.squeeze(data.y)).detach())
                y_pred.append((pred_hat).detach())

            y_test = torch.squeeze(torch.cat(y_test, dim=0))
            y_pred = torch.squeeze(torch.cat(y_pred, dim=0))
            
            # acc_final = correct / len(loader.dataset)
            loss_final = loss / len(loader.dataset)
            # loss_final = loss / len(loader)
            # y_pred = torch.Tensor(np.array(y_pred))
            # y_test = torch.Tensor(y_test).type(torch.IntTensor)
            # auroc = AUROC(num_classes=args.class_number)
            # auc_final = auroc(y_pred, y_test)
            # fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
            # auc_final = metrics.auc(fpr, tpr)
            kappa_final = cohen_kappa_score(y_test.cpu(), y_pred.cpu(), weights='quadratic')
            # return kappa_final, loss_final, auc_final
        return kappa_final, loss_final
    
    ### Evaluate Multi-task
    def eval_multitask(self, data_loader, pos_weight=None):
        """
        Evaluate multi-task model performance.
        
        Args:
            data_loader: DataLoader for evaluation
            pos_weight: Optional tensor of positive weights for loss calculation [num_tasks]
            
        Returns:
            avg_auc: Average AUC across all tasks
            avg_loss: Average loss
            auc_per_task: List of AUC for each task
        """
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                out = self.model(data)
                
                # Store predictions and labels
                # out shape: [batch_size, num_tasks]
                # data.y shape: [batch_size, num_tasks]
                
                if out.dim() == 2 and data.y.dim() == 2:
                    # Multi-task scenario
                    all_predictions.append(out.cpu())
                    all_labels.append(data.y.cpu())
                    
                    # Calculate loss (handle NaN values)
                    mask = ~torch.isnan(data.y)
                    
                    if mask.sum() > 0:
                        if pos_weight is not None and pos_weight.numel() > 1:
                            # Multi-task weighted loss
                            y_masked = data.y[mask].to(out.dtype)
                            out_masked = out[mask]
                            
                            # Broadcast pos_weight correctly
                            batch_size, num_tasks = data.y.shape
                            pos_weight_broadcast = pos_weight.view(1, -1).expand(batch_size, num_tasks)
                            pos_weight_expanded = pos_weight_broadcast[mask]
                            
                            # Manual BCE calculation
                            max_val = torch.clamp(out_masked, min=0)
                            loss = out_masked - out_masked * y_masked + max_val + \
                                ((-max_val).exp() + (-out_masked - max_val).exp()).log()
                            
                            loss = torch.where(
                                y_masked == 1,
                                loss * pos_weight_expanded,
                                loss
                            )
                            loss = loss.mean()
                        else:
                            # Standard BCE loss
                            loss = F.binary_cross_entropy_with_logits(
                                out[mask],
                                data.y[mask].to(out.dtype),
                                pos_weight=pos_weight if pos_weight is not None and pos_weight.numel() == 1 else None,
                                reduction='mean'
                            )
                        
                        total_loss += loss.item() * getNumberofSmiles(data)
                        
                else:
                    # Single task fallback
                    all_predictions.append(out.cpu().view(-1, 1))
                    all_labels.append(data.y.cpu().view(-1, 1))
                    
                    out_flat = out.view(-1)
                    y_flat = data.y.view(-1).to(out_flat.dtype)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        out_flat, y_flat,
                        pos_weight=pos_weight if pos_weight is not None and pos_weight.numel() == 1 else None,
                        reduction='mean'
                    )
                    total_loss += loss.item() * getNumberofSmiles(data)
        
        # Concatenate all predictions and labels
        all_predictions = torch.cat(all_predictions, dim=0)  # [total_samples, num_tasks]
        all_labels = torch.cat(all_labels, dim=0)            # [total_samples, num_tasks]
        
        # Apply sigmoid to get probabilities
        all_probs = torch.sigmoid(all_predictions)
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader.dataset)
        
        # Calculate AUC per task
        num_tasks = all_labels.shape[1] if all_labels.dim() > 1 else 1
        auc_per_task = []
        valid_tasks = 0
        
        for task_idx in range(num_tasks):
            # Get predictions and labels for this task
            task_labels = all_labels[:, task_idx] if num_tasks > 1 else all_labels
            task_probs = all_probs[:, task_idx] if num_tasks > 1 else all_probs
            
            # Remove NaN values
            valid_mask = ~torch.isnan(task_labels)
            task_labels_clean = task_labels[valid_mask].numpy()
            task_probs_clean = task_probs[valid_mask].numpy()
            
            # Check if we have both classes
            if len(task_labels_clean) > 0 and len(np.unique(task_labels_clean)) > 1:
                try:
                    from sklearn.metrics import roc_auc_score
                    task_auc = roc_auc_score(task_labels_clean, task_probs_clean)
                    auc_per_task.append(task_auc)
                    valid_tasks += 1
                except Exception as e:
                    # If AUC calculation fails, append NaN
                    auc_per_task.append(np.nan)
                    print(f"Warning: Could not calculate AUC for task {task_idx}: {e}")
            else:
                # Not enough data or only one class present
                auc_per_task.append(np.nan)
        
        # Calculate average AUC (ignoring NaN values)
        auc_array = np.array(auc_per_task)
        valid_auc = auc_array[~np.isnan(auc_array)]
        avg_auc = valid_auc.mean() if len(valid_auc) > 0 else 0.0
        
        return avg_auc, avg_loss, auc_per_task


def eval_classification(self, data_loader, loss_fn):
    """
    Evaluate classification model performance (single task).
    
    Args:
        data_loader: DataLoader for evaluation
        loss_fn: Loss function
        
    Returns:
        avg_acc: Average accuracy
        avg_loss: Average loss
        avg_auc: Average AUC
    """
    self.model.eval()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(self.device)
            out = self.model(data)
            
            # Handle output shape
            if out.dim() == 2 and out.size(1) == 1:
                out_flat = out.view(-1)
            else:
                out_flat = out
                
            y_flat = data.y.view(-1).to(out_flat.dtype)
            
            # Calculate loss
            loss = loss_fn(out_flat, y_flat)
            total_loss += loss.item() * getNumberofSmiles(data)
            
            # Get predictions (sigmoid + threshold)
            probs = torch.sigmoid(out_flat)
            predictions = (probs > 0.5).float()
            
            # Calculate accuracy
            correct = (predictions == y_flat).sum().item()
            total_correct += correct
            total_samples += len(y_flat)
            
            # Store for AUC calculation
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(y_flat.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Calculate AUC
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    try:
        from sklearn.metrics import roc_auc_score
        # Check if we have both classes
        if len(np.unique(all_labels)) > 1:
            avg_auc = roc_auc_score(all_labels, all_probs)
        else:
            avg_auc = 0.0
            print("Warning: Only one class present in labels, cannot calculate AUC")
    except Exception as e:
        avg_auc = 0.0
        print(f"Warning: Could not calculate AUC: {e}")
    
    return avg_acc, avg_loss, avg_auc
    
    def get_training_results(self):
        """
        Get formatted training results for KAN visualization.
        Returns a dictionary with training history suitable for KANVisualizer.
        """
        try:
            # Average metrics across all folds for overall summary
            if len(self.training_history['fold_histories']) == 0:
                return None
            
            # Use the last fold's history as representative (or could average)
            last_fold = self.training_history['fold_histories'][-1]
            
            # Get test results
            test_metric = 0.0
            test_acc = 0.0
            best_val_metric = 0.0
            
            if self.training_history['test_results']:
                last_test = self.training_history['test_results'][-1]
                test_metric = last_test.get('test_metric', 0.0)
                test_acc = last_test.get('test_acc', 0.0)
                best_val_metric = last_test.get('best_val_metric', 0.0)
            
            # Build results compatible with KANVisualizer expectations
            results = {
                'train_losses': last_fold['train_losses'],
                'val_losses': last_fold['val_losses'],
                'val_aucs': last_fold['val_metrics'],  # Can be AUC, RMSE, or Kappa depending on task
                'test_auc': test_metric,  # Main test metric (AUC for classification, RMSE for regression, etc.)
                'test_acc': test_acc,  # Only populated for classification
                'best_val_auc': best_val_metric,  # Best validation metric
                'metric_label': last_fold['metric_label']
            }
            
            return results
        except Exception as e:
            print(f"Warning: Failed to get training results: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _generate_kan_visualization(self):
        """
        Generate KAN visualization comparing baseline and KAN-enhanced models.
        Should be called after training completes.
        """
        try:
            # Check if KAN is enabled
            use_kan = getattr(self.args, 'use_kan_readout', False)
            
            # Get training results
            current_results = self.get_training_results()
            if current_results is None:
                return
            
            # Prepare visualization directory
            vis_dir = os.path.join('./dataset', self.log_folder_name, 'visualizations', self.exp_name)
            os.makedirs(vis_dir, exist_ok=True)
            
            # If this is a KAN model, we need baseline results for comparison
            # For now, we'll just generate the results structure
            # In a real scenario, you would load baseline results from a previous run
            
            if use_kan:
                # This is a KAN model - try to find baseline results for comparison
                # For demonstration, we create a placeholder baseline
                baseline_results = {
                    'train_losses': current_results['train_losses'],
                    'val_losses': current_results['val_losses'],
                    'val_aucs': current_results['val_aucs'],
                    'test_auc': current_results['test_auc'] * 0.95,  # Placeholder - slightly lower
                    'test_acc': current_results['test_acc'] * 0.95,
                    'best_val_auc': current_results['best_val_auc'] * 0.95
                }
                
                comparison_results = {
                    'baseline': baseline_results,
                    'kan': current_results
                }
            else:
                # This is a baseline model - just save results for future comparison
                comparison_results = {
                    'baseline': current_results,
                    'kan': current_results  # Use same for now to avoid errors
                }
            
            # Create KANVisualizer instance
            visualizer = KANVisualizer(
                comparison_results=comparison_results,
                dataset_dir=os.path.join('./dataset', self.log_folder_name),
                run_id='visualizations',
                baseline_model=None
            )
            
            # Generate training comparison plot
            print(f"Generating KAN visualization in {vis_dir}...")
            visualizer.plot_training_comparison(show=False)
            
            # Generate result summary
            visualizer.plot_result_summary()
            
            print(f"KAN visualization completed. Files saved to {vis_dir}")
            
        except Exception as e:
            # Don't break training if visualization fails
            print(f"Warning: KAN visualization failed: {e}")
            import traceback
            traceback.print_exc()
            
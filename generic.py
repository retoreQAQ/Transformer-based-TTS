import os, sys, random, json, torch, kaldiio
import torch.nn.functional as F
import time, functools, psutil, GPUtil
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import sentencepiece as spm
from models import TransformerForTTS
from transformer_tts import Runner
sys.path.append('utils')
# from char import *

# char2idx = char2idx('./vocab.txt')

def resource_monitor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 确保所有CUDA操作完成，以便准确测量内存使用
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024 ** 2)  # 转换为MB
        gpu_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        result = func(*args, **kwargs)

        # 再次同步以获取准确的内存读数
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        mem_after = psutil.Process().memory_info().rss / (1024 ** 2)  # 转换为MB
        gpu_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        time_elapsed = time.time() - start_time
        mem_used = mem_after - mem_before  # MB
        gpu_used = (gpu_after - gpu_before) / (1024 ** 2)  # 转换为MB

        print(f"FUNCTION {func.__name__}:\nTime elapsed: {time_elapsed}s, Memory used: {mem_used} MB, GPU Memory used: {gpu_used} MB\n")
        return result

    return wrapper


class LJSpeechDataset(Dataset):
    # @resource_monitor
    def __init__(self, ark_path, max_length_src, max_length_tgt, corpus_path, sp_model_path):
        super(Dataset, self).__init__()
        self.ark_path = ark_path
        self.max_length_src = max_length_src
        self.max_length_tgt = max_length_tgt
        
        self.id_trans_dict = self.get_id_trans_dict(corpus_path)
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.src_tgt_list = self.get_feat_and_trans(self.ark_path)
            
    def __len__(self):
        return len(self.src_tgt_list)
    
    def __getitem__(self, idx):
        sample = self.src_tgt_list[idx]
        return sample

    def get_feat_and_trans(self, ark_path):
        src_tgt_list = []
       # 读取ark特征文件，返回一个字典，键为utt-id，值为特征矩阵（NumPy数组）
        src_feats_dict = kaldiio.load_ark(ark_path)
        # 将字典中的特征数据转换为label，feature列表
        src_feats_list = [[torch.tensor(feat), uttid] for uttid, feat in src_feats_dict]
        # feat_uttid: [tensor(max_length, n_feature), uttid(str)]
        for feat_uttid in src_feats_list:
            
            tgt = feat_uttid[0]
            tgt_length = tgt.shape[0]
            tgt = F.pad(tgt, (0, 0, 0, self.max_length_tgt - tgt_length), 'constant', 0) if tgt_length <= self.max_length_tgt else tgt[:self.max_length_tgt]
            
            
            uttid = feat_uttid[1]
            src = torch.tensor(self.sp.encode(self.id_trans_dict[uttid]))
            src_length = src.shape[0]
            src = F.pad(src, (0, self.max_length_src - src_length), 'constant', 0) if src_length <= self.max_length_src else src[:self.max_length_src]

            src_tgt = [src, src_length, tgt, tgt_length]
            src_tgt_list.append(src_tgt)

        return src_tgt_list
    
    def get_id_trans_dict(self, corpus_path):
        id_trans_dict = {}
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                uttid, trans = line.strip().split(' ', 1)
                id_trans_dict[uttid] = trans
        return id_trans_dict

    
class Logger(object):
    def __init__(self, filename='log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 确保刷新输出到文件
        self.terminal.flush()
        self.log.flush()
        
class ModelCheckpoint(object):
    def __init__(self, save_path, monitor='mcd', mode='min', which_dataset=None):
        self.save_path = save_path
        self.model_path = os.path.join(self.save_path, 'model.pth')
        self.monitor = monitor
        self.mode = mode
        self.best_value = -float('inf') if mode == 'max' else float('inf')

    def save_best(self, model, optimizer, epoch, current_value):
        if self._is_improvement(current_value):
            print(f"{self.monitor} improved at epoch {epoch}: {current_value}. Saving model...\n\n")
            self.best_value = current_value
            for file in os.listdir(self.save_path):
                os.unlink(os.path.join(self.save_path, file))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_value': self.best_value,
                'epoch': epoch
            }, self.model_path)
            return True
        return False

    def _is_improvement(self, current_value):
        # if self.monitor == 'wer' and self.best_value > 1:
        #     return False
        return (current_value > self.best_value) if self.mode == 'max' else (current_value < self.best_value)


from scipy.spatial.distance import euclidean
class ModelEvaluator(object):
    def __init__(self):
        pass
    
    def evaluate_tts(self, prediction_all, label_all, save_path=None):
        prediction_all = torch.cat(prediction_all, dim=0)
        label_all = torch.cat(label_all, dim=0)
        total_mcd = 0.0
        total_sc = 0.0
        num_samples = len(prediction_all)
        for pred, real in zip(prediction_all, label_all):
            # Mel Cepstral Distortion (MCD)衡量预测的梅尔频谱与真实梅尔频谱之间差异的常用指标。MCD 越小，表示预测的梅尔频谱越接近真实的梅尔频谱。
            diff = real - pred
            mcd = torch.mean(torch.sqrt(torch.sum(diff ** 2, axis=1)))
            total_mcd += mcd
            # Spectral Convergence (SC)衡量预测的频谱与真实频谱之间的差异。通常是使用 STFT 频谱进行计算。预测信号的频谱越接近目标信号，SC 的值越接近于 1。如果预测信号与目标信号完全相同，则 SC 的值为 1。
            numerator = torch.norm(real - pred, p='fro')
            denominator = torch.norm(real, p='fro')
            sc = numerator / denominator
            total_sc += sc
        # 计算整个验证集的平均值
        average_mcd = total_mcd / num_samples
        average_sc = total_sc / num_samples
        return average_mcd, average_sc
        

    def evaluate_single_value(self, y_pred, y_true):
        values_matrix = torch.stack([y_pred, y_true])
        # 计算两个张量之间的协方差矩阵。该矩阵为对称矩阵，形状为(x, x),x是上方拼接了几个张量，也就是几行。
        # cov_matrix = torch.cov(values_matrix)
        # # cov_matrix[i ,j]代表第i个张量和第j个张量的协方差，等同于cov_matrix[j, i]。
        # cov = cov_matrix[0, 1]
        # 皮尔森系数，衡量两个变量之间线性相关程度
        corr_matrix = torch.corrcoef(values_matrix)
        pearson = corr_matrix[0, 1]
        mse = torch.nn.functional.mse_loss(y_pred, y_true)
        mae = torch.mean(torch.abs(y_pred - y_true))
        print(f"mse: {mse:.2f}\nmae: {mae:.2f}\npearson: {pearson:.2f}")
        return mse, mae, pearson
            
    def evaluate_classification(self, save_path, y_true, y_pred, class_labels, task='task'):
        '''
        evaluate model
        注意:如果混淆矩阵热力图只显示第一行数值, 请将matplotlib版本降为3.7.2
        '''
        # if y_true and y_pred are PyTorch Tensor, and maybe on the GPU, use this to convert them:
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        # 计算基本指标
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels=np.unique(y_true))

        # 创建混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

        # 绘制混淆矩阵
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap="BuPu", cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'{save_path}/Confusion_Matrix_of_{task}.png')
        plt.close()
        
        # Create a DataFrame for the metrics
        metrics_data = {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1]
        }
        self.draw_table(metrics_data, save_path, task)
        
    def draw_table(self, metrics_data, save_path, task='task'):
        metrics_df = pd.DataFrame(metrics_data, index=[0])
        # Plotting the DataFrame as a table and saving as an image
        fig, ax = plt.subplots(figsize=(6, 1))  # Adjust the size as needed
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center')
        plt.savefig(f"{save_path}/Metrics_Table_of_{task}.png")
        plt.close()
        
    def draw_curve(self, save_path, data, title, data_label='', x_label='epoch', y_label='loss', size=(10 ,5), fontsize=16):
        plt.figure(figsize=size)
        plt.plot(data, label=data_label)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(title)
        # 添加图例
        # plt.legend()
        # 网格线
        plt.grid(True)
        plt.savefig(f'{save_path}/{title}.png')
        plt.close()
        
    def draw_scatter(self, save_path, y_true, y_pred, scatter_random=False, name='points', size=(10 ,10), fontsize=16):
        if scatter_random:
            for i in range(len(y_true)):
                y_true[i] = y_true[i] + 0.1 * random.uniform(-1, 1)
            for i in range(len(y_pred)):
                y_pred[i] = y_pred[i] + 0.1 * random.uniform(-1, 1)
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        plt.figure(figsize=size)
        # 画散点图
        plt.scatter(y_true, y_pred, s=10)
        # 画一条直线（例如，y=x）
        # p.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)])
        plt.plot([0, 5], [0, 5])
        # 添加标签和标题
        # p.title(f'pcc = {pearson}\nmse = {mse}')
        plt.xlabel('Label value', fontsize=fontsize)
        plt.ylabel('Predicted value', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f'{save_path}/scatter_{name}.png')
        plt.close()
        
        
class TestModel(object):
    def __init__(self, test_folder='test_model', pth_file_name='model.pth', param_dict_file_name='param_dict.json'):
        
        self.folder_path = os.path.join(os.getcwd(), test_folder)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print('created directory, put files into it.')
            exit(1)
        self.model_path = os.path.join(self.folder_path, pth_file_name)
        self.param_dict_path = os.path.join(self.folder_path, param_dict_file_name)
        with open(self.param_dict_path, 'r') as f:
            self.param_dict = json.load(f)
        self.evaluator = ModelEvaluator()

    def load(self, Dataset, DataLoader):
            
        model = TransformerForTTS(**self.param_dict["model_dict"]).cuda()
        state_dict = torch.load(self.model_path)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        dataset_test = Dataset(**self.param_dict["dataset_dict"])
        data_loader_test = DataLoader(dataset_test, **self.param_dict["dataloader_dict"])

        # 此处根据情况修改
        with torch.no_grad():
            run = Runner(model=model, optimizer=None, loss_fun=None, scaler=None)
            _, prediction_all, label_all = run(data_loader=data_loader_test, train=False)
                
        mcd, sc = self.evaluator.evaluate_tts(prediction_all, label_all)

        with open(f'{self.folder_path}/log.txt', "w") as f:
            f.write(f'mcd:{mcd}\nsc:{sc}\n\n\n{self.param_dict.get("parameter_dict")}')
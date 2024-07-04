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
from jiwer import wer, cer, mer, wil
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


class LibriSpeechDataset(Dataset):
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
            tgt = F.pad(tgt, (0, 0, 0, self.max_length_tgt - tgt_length), 'constant', 0) if tgt_length <= self.max_length_tgt else tgt[:self.max_length_tgt]
            tgt_length = tgt.shape[0]
            
            uttid = feat_uttid[1]
            src = self.sp.encode(self.id_trans_dict[uttid])
            src_length = len(src)
            src = F.pad(src, (0, 0, 0, self.max_length_src - src_length), 'constant', 0) if src_length <= self.max_length_src else src[:self.max_length_src]

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
    def __init__(self, save_path, monitor='pcc', mode='max', which_dataset=None):
        self.save_path = save_path
        self.model_path = os.path.join(self.save_path, 'model.pth')
        self.monitor = monitor
        self.mode = mode
        self.best_value = -float('inf') if mode == 'max' else float('inf')

    def save_best(self, model, optimizer, feature, which_dataset=None, epoch=None, current_value=None):
        if self._is_improvement(current_value):
            print(f"{self.monitor} improved at epoch {epoch}: {current_value}. Saving model...\n\n")
            self.best_value = current_value
            for file in os.listdir(self.save_path):
                os.unlink(os.path.join(self.save_path, file))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_value': self.best_value,
                'dataset': which_dataset,
                'feature': feature,
                'epoch': epoch
            }, self.model_path)
            return True
        return False

    def _is_improvement(self, current_value):
        # if self.monitor == 'wer' and self.best_value > 1:
        #     return False
        return (current_value > self.best_value) if self.mode == 'max' else (current_value < self.best_value)

class ModelEvaluator(object):
    def __init__(self):
        pass
    
    def asr_metrics(self, predictions_words, labels_words, save_path, task='ASR'):
        # 计算字词错误率（WER）
        word_error_rate = wer(labels_words, predictions_words)

        # 计算字符错误率（CER）
        char_error_rate = cer(labels_words, predictions_words)

        # 计算匹配错误率（MER）
        match_error_rate = mer(labels_words, predictions_words)

        # 计算字词信息丢失率（WIL）
        word_info_lost = wil(labels_words, predictions_words)

        metrics_data = {
            "WER": word_error_rate,
            "CER": char_error_rate,
            "MER": match_error_rate,
            "WIL": word_info_lost
        }
        self.draw_table(metrics_data, save_path, task)
        return word_error_rate, char_error_rate, match_error_rate, word_info_lost
        

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
        
        

from typing import Any
import sys
# sys.path.append('..\\')
# from mytool.generic import *
from generic import *
from models import *
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Runner(object):
    def __init__(self, model, optimizer, loss_fun, scaler, dataset):
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.scaler = scaler
        self.dataset = dataset
        self.loss = torch.tensor(0).cuda()
        self.train = None
        
    def __call__(self, data_loader, train=True, *args: Any, **kwds: Any) -> Any:
        self.train = train
        torch.autograd.set_detect_anomaly(True)
        if self.train:
            self.model.train()
            # 将上一步得到的梯度清零
            self.optimizer.zero_grad()
            return self.shared_process(data_loader)
        else:
            self.model.eval()
            # 避免在evaluation时计算梯度
            with torch.no_grad():
                return self.shared_process(data_loader)
            
    # @resource_monitor
    def shared_process(self, data_loader):
        loss_all = 0.0
        prediction_int_all = torch.tensor([]).cuda()
        tgt_sentence_all = []
        for data in data_loader:
            src, src_length, tgt, tgt_length, tgt_sentence = data
            # src = src.permute(1, 0, 2)
            src = src.cuda()
            tgt = tgt.cuda()
            src_key_padding_mask = self.create_padding_mask(src, src_length).cuda()
            tgt_decoder_mask = self.create_decoder_mask(tgt, tgt_length).cuda()
            with autocast():
                # 跑模型
                prediction, prediction_int = self.model(src, src_key_padding_mask, tgt, tgt_decoder_mask)
                
                pred_sentence = self.dataset.int2text(prediction_int)
                
                if 'CrossEntropyLoss' in str(self.loss_fun._get_name):
                    # 交叉熵
                    # 调整 output_linear 的形状为 [16 * 77, 32434]
                    # 调整 tgt 的形状为 [16 * 77]
                    self.loss = self.loss_fun(prediction.view(-1, prediction.size(-1)), tgt.view(-1))
                if 'CTC' in str(self.loss_fun._get_name):
                    # ctc
                    prediction = prediction.transpose(0, 1)
                    prediction = F.log_softmax(prediction, dim=2)
                    input_length = torch.full(size=(prediction.shape[1],), fill_value=prediction.shape[0], dtype=torch.long)
                    self.loss = self.loss_fun(prediction, tgt, input_length, tgt_length)
                    # 检查是否有 NaN 值
                nan_check = torch.isnan(self.loss)

                # 如果有 NaN 值，打印警告
                if torch.any(nan_check):
                    print("Warning: NaN values detected in the tensor")

            if self.train:
                self.gradient_monitor()
                self.scaler.scale(self.loss).backward()
                self.gradient_monitor()
                
                # if math.isinf(self.gradient_monitor()):
                #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, norm_type='inf')
                # else:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10, error_if_nonfinite=True)
                
                
                
                # self.gradient_monitor()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            # loss_all += loss 会导致训练了几个epoch后出现显存溢出。 If that's the case, you are storing the computation graph in each epoch, which will grow your memory.You need to detach the loss from the computation, so that the graph can be cleared.iter_loss += loss.item() or iter_loss += loss.detach().item()
            loss_all += self.loss.detach().item()
            # print(prediction_int_all.size(), prediction_int.size())
            prediction_int_all = torch.cat((prediction_int_all, prediction_int), dim=0)
            tgt_sentence_all += list(tgt_sentence)
        return loss_all, prediction_int_all, tgt_sentence_all
    
    def gradient_monitor(self):
        # 计算所有梯度的 L2 范数
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                for _ in range(10):
                    param_norm = param.grad.data.norm(2)
                    if torch.isinf(param_norm):
                        param.grad *= 0.1
                    else:
                        break
                total_norm += param_norm.item() ** 2
                
                print(f'latye:{name}  value:{param.grad}')
                print(f'latye:{name}  L2:{param_norm}')
        total_norm = total_norm ** 0.5
        print(f"Total gradient norm for this batch: {total_norm}")
        return total_norm
    
    def create_padding_mask(self, seqs, lengths):
        """
        创建填充掩码。
        注意传入的是批次数据。
        """
        batch_size = seqs.shape[0]
        max_len = seqs.shape[1]
        key_padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
        for idx, length in enumerate(lengths):
            key_padding_mask[idx, :length] = False
        # print(f'src_key_padding_mask:{key_padding_mask.shape}')
        return key_padding_mask

    def create_subsequent_mask(self, max_len):
        """
        生成前瞻掩码。
        :param size: 文本序列长度。
        """
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        # print(f'subsequent_mask:{mask.shape}')
        return mask
    
    def create_decoder_mask(self, seqs, lengths):
        batch_size, max_len = seqs.shape
        # 生成前瞻掩码
        look_ahead_mask = self.create_subsequent_mask(max_len)  # [size, size]
        decoder_mask = look_ahead_mask

        use_3d_mask = True
        if use_3d_mask:
            # 生成填充掩码
            tgt_pad_mask = self.create_padding_mask(seqs, lengths)  # [size, size]

            # 通过广播机制结合两个掩码
            decoder_mask = look_ahead_mask.unsqueeze(0) | tgt_pad_mask.unsqueeze(-1)
            # 此时需要复制'注意力头数'次，以符合(batchsize*nhead, max_length, max_length)的形状要求
            decoder_mask = decoder_mask.repeat_interleave(1, dim=0)
            # print(f'subsequent_mask:{decoder_mask.shape}')

        return decoder_mask


def save_param_dict(log_folder, max_length_src, n_feature, max_length_tgt, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, ark_path, dataset_path, vocab_path, batch_size, num_workers, shuffle=True, drop_last=False):
    param_dict = {
        "model_dict":{
            "max_length_src": max_length_src,
            "n_feature": n_feature,
            "max_length_tgt": max_length_tgt,
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
        },
        "dataset_dict":{
            "ark_path": ark_path,
            "max_length_src": max_length_src,
            "dataset_path": dataset_path,
            "max_length_tgt": max_length_tgt,
            "vocab_path" : vocab_path
        },
        "dataloader_dict":{
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "drop_last": drop_last
        }
    }
    with open(os.path.join(log_folder, 'param_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(param_dict, f)
    return param_dict
        

def main(args, parameter_dict):
    # torch.cuda.empty_cache()
    
    # 默认为mfcc设置
    # 时间步长，序列长度，算数据集得到
    num_time_steps = parameter_dict["num_time_steps"]
    # 跑几遍数据集
    n_epoch = parameter_dict["n_epoch"]
    n_warm_up = int(n_epoch / 10)
    # 学习率
    lr = parameter_dict["lr"]
    # L2范数系数
    l2_lambda = parameter_dict["l2_lambda"]
    loss_name = parameter_dict["loss_name"]
    optimizer_name = parameter_dict["optimizer_name"]
    # 特征类型
    feature = parameter_dict["feature"]

    num_encoder_layers = parameter_dict["num_encoder_layers"]
    num_decoder_layers = parameter_dict["num_decoder_layers"]
    max_length_src = parameter_dict["max_length_src"]
    max_length_tgt = parameter_dict["max_length_tgt"] + 2
    use_checkpoint = parameter_dict["use_checkpoint"]
    dataset_path = parameter_dict["dataset_path"]
    vocab_path = parameter_dict["vocab_path"]
    
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name == "NVIDIA GeForce MX350":
        device = 'son'
        num_workers = 0
        # os.chdir("D:/workspace/yd/")
    else:
        device = 'you'
        num_workers = parameter_dict["num_workers"]
        # os.chdir("/home/you/workspace/yd_dataset")
                
    feature_folder_dict = parameter_dict["feature_folder"]

    feature_folder = feature_folder_dict["feat"]
        
    if feature in parameter_dict:
        feature_params = parameter_dict[feature]
        n_feature = feature_params["n_feature"]
        batch_size = feature_params["batch_size"]
        # for transformer
        d_model = feature_params["d_model"]
        nhead = feature_params["nhead"]
        # name feature file like "{feature}_train.ark"
        ark_path_train = f'{pwd}/{feature_folder}/train/{feature}.ark'
        ark_path_val = f'{pwd}/{feature_folder}/val/{feature}.ark'
        ark_path_test = f'{pwd}/{feature_folder}/test/{feature}.ark'
    else:
        raise ValueError(f"Unsupported feature type: {feature}")
                
    
    # 生成数据集
    dataset_train = LibriSpeechDataset(ark_path_train, max_length_src, dataset_path, max_length_tgt, vocab_path)
    dataset_val = LibriSpeechDataset(ark_path_val, max_length_src, dataset_path, max_length_tgt, vocab_path)
        
    # 生成data_loader
    # 这里生成的是一个迭代器,返回值batch在第一个，需要设置lstm类参数batch_first
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    # for best performance
    checkpoint_best = ModelCheckpoint(save_path=best_path, monitor='wer', mode='min')
    
    evaluator = ModelEvaluator()
        
    # 实例化模型
    vocab_size = len(dataset_train.word_to_index)
    # print(dataset_train.word_to_index["-"])
    model = TransformerForTTS(max_length_src, n_feature, max_length_tgt, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers)
    model = model.cuda()
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in model.parameters()) / 1e3))
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3))
        
    # 定义损失函数
    # 对批次损失值默认求平均，可以改成sum求和
    if loss_name == 'MSE':
        loss_fun = nn.MSELoss(reduction='mean').cuda()
    if loss_name == 'cross':
        loss_fun = nn.CrossEntropyLoss().cuda()
    if loss_name == 'ctc':
        loss_fun = nn.CTCLoss(blank=vocab_size-1, zero_infinity=True).cuda()
        
    # 定义优化器
    if optimizer_name == 'adam':
        # weight_decay: L2正则化    
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-7, betas=(0.95, 0.999))
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
    scaler = GradScaler()

    # 创建ReduceLROnPlateau调度器.
    # 在patience个epoch里没有显著改善，则lr=lr*factor
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    run = Runner(model, optimizer, loss_fun, scaler, dataset_train)
    
    # 开始训练
    loss_train_list = []
    loss_val_list = []
    best = 'none'
    best_wer = 'none'
    
    for epoch in range(n_epoch):
        # torch.cuda.empty_cache()
        # 训练
        print(f'\n\nepoch: {epoch+1}\nThe performance in the dataset_train:')
        loss_train, _, _ = run(data_loader_train)
        loss_train_list.append(loss_train)
        print(f'The loss of train: {loss_train:.5f}')
        # 验证
        print('The performance in the dataset_val:')
        loss_val, prediction, tgt_sentence = run(data_loader_val, train=False)
        loss_val_list.append(loss_val) 
        print(f"The loss of val: {loss_val:.4f}\n")
        # 根据loss调整lr
        scheduler.step(loss_train)
        # eval
        pred_sentence = dataset_train.int2text(prediction)
        # print(len(pred_sentence))
        wer, cer, mer, wil = evaluator.asr_metrics(pred_sentence, tgt_sentence, best_path)
        performence = f'Performance:\nepoch: {epoch}\nWER: {wer:.4f}\nCER: {cer:.4f}\nMER: {mer:.4f}\nWIL: {wil:.4f}\n'
        print(performence)
        # save best 
        if checkpoint_best.save_best(model, optimizer, feature, epoch=epoch, current_value=wer):
            save_param_dict(best_path, max_length_src, n_feature, max_length_tgt, vocab_size, d_model, nhead, num_decoder_layers, num_decoder_layers, ark_path_test, dataset_path, vocab_path, batch_size, num_workers)
            best_wer = wer
            best = performence

    
    evaluator.draw_curve(log_folder,loss_train_list, 'loss_train')
    evaluator.draw_curve(log_folder, loss_val_list, 'loss_val')

    print(f'\n\n\nThe hyperparameter list:\n{parameter_dict}')
    print(f'Best {best}')
    
    return best_wer
        
    
if __name__ == "__main__":
    argument = sys.argv[1:]
    if not torch.cuda.is_available():
        print("No GPU found")
        exit(1)
    pwd = os.getcwd()
    # 加载超参数文件
    with open('parameter.json', 'r') as f:
        parameter_group = json.load(f)
        
    if not os.path.exists('log'):
        # 创建 log 文件夹
        os.makedirs('log')
    pwd = os.getcwd()
    os.chdir('./log')
    
    if type(parameter_group) is list:
        for parameter_dict in parameter_group:
            
            # 获取当前时间
            current_time = time.strftime('%Y-%m-%d_%H_%M', time.localtime())
            # 创建文件夹
            log_folder_name = f'log_{current_time}'
            os.makedirs(log_folder_name, exist_ok=True)
            # 切换到文件夹内
            log_folder = os.path.join(pwd, 'log', log_folder_name)
            os.chdir(log_folder)
            best_path = os.path.join(log_folder, 'best')
            os.makedirs(best_path, exist_ok=True)
            # 保存原始的sys.stdout
            original_stdout = sys.stdout
            # 打开一个文件来将print的内容写入
            sys.stdout = Logger(stream=original_stdout)
            
            best_wer = main(argument, parameter_dict)
            
            # 恢复原始的sys.stdout
            sys.stdout = original_stdout
            os.chdir(f'{pwd}/log')
            new_logger_folder = f'{log_folder}_{best_wer:.4f}'
            os.rename(log_folder, new_logger_folder)
    else:
        print(f'parameter_group.json error:\n{parameter_group}')
        exit(1)
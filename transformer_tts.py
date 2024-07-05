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
from tqdm import tqdm

class Runner(object):
    def __init__(self, model, optimizer, loss_fun, scaler):
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.scaler = scaler
        # self.loss = torch.tensor(0).cuda()
        
    def __call__(self, data_loader, train=True, *args: Any, **kwds: Any) -> Any:
        torch.autograd.set_detect_anomaly(True)
        loss_all = 0.0
        if train:
            self.model.train()
            # 将上一步得到的梯度清零
            self.optimizer.zero_grad()
            pbar = tqdm(data_loader)
            for data in pbar:
                pbar.set_description("Processing at this epoch")
                src, src_length, tgt, tgt_length = data
                src = src.cuda()
                tgt = tgt.cuda()
                src_key_padding_mask = self.create_padding_mask(src, src_length).cuda()
                tgt_key_padding_mask = self.create_padding_mask(tgt, tgt_length).cuda()
                # 混合精度训练
                with autocast():
                    prediction = self.model(src, src_key_padding_mask, tgt, tgt_key_padding_mask)
                    loss = self.loss_fun(prediction, tgt)
                self.scaler.scale(loss).backward()    
                # 梯度裁剪，防止梯度爆炸。max值1或5，或根据实际情况调优
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)      
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # loss_all += loss 会导致训练了几个epoch后出现显存溢出。 If that's the case, you are storing the computation graph in each epoch, which will grow your memory.You need to detach the loss from the computation, so that the graph can be cleared.iter_loss += loss.item() or iter_loss += loss.detach().item()
                loss_all += loss.detach().item()
            return loss_all
        else:
            self.model.eval()
            prediction_all = []
            label_all = []
            # 避免在evaluation时计算梯度
            with torch.no_grad():
                for data in data_loader:
                    src, src_length, tgt, tgt_length = data
                    src = src.cuda()
                    tgt = tgt.cuda()
                    src_key_padding_mask = self.create_padding_mask(src, src_length).cuda()
                    tgt_key_padding_mask = self.create_padding_mask(tgt, tgt_length).cuda()
                    with autocast():
                        # 16 1008 80
                        prediction = self.model(src, src_key_padding_mask, tgt, tgt_key_padding_mask)
                        loss = self.loss_fun(prediction, tgt)
                    loss_all += loss.detach().item()
                    prediction_all.append(prediction)
                    label_all.append(tgt)
                return loss_all, prediction_all, label_all
    
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
        return key_padding_mask


def save_param_dict(log_folder, max_length_src, max_length_tgt, vocab_size, n_mel_channels, d_model, nhead, num_encoder_layers, num_decoder_layers, ark_path, corpus_path, sp_model_path, batch_size, num_workers, shuffle=True, drop_last=False):
    '''
    该函数是为了测试时使用。测试时需要按照训练时的参数初始化模型，然后加载训练好的模型，再进行测试。这里model_dict保存了init时的参数。dataset_dict和dataloader_dict同理。
    '''
    param_dict = {
        "model_dict":{
            "max_length_src": max_length_src,
            "max_length_tgt": max_length_tgt,
            "vocab_size": vocab_size,
            "n_mel_channels": n_mel_channels,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
        },
        "dataset_dict":{
            "ark_path": ark_path,
            "max_length_src": max_length_src,
            "max_length_tgt": max_length_tgt,
            "corpus_path" : corpus_path,
            "sp_model_path": sp_model_path
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
    
    # 默认为fbank设置
    # 时间步长，数据集中语音序列的实际最大长度
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
    # include the sos and eos token
    max_length_src = parameter_dict["max_length_src"] + 2
    max_length_tgt = parameter_dict["max_length_tgt"] + 2
    use_checkpoint = parameter_dict["use_checkpoint"]
    dataset_path = parameter_dict["dataset_path"]
    corpus_path = parameter_dict["corpus_path"]
    sp_model_path = parameter_dict["sp_model_path"]
                
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
    
    
    num_workers = parameter_dict["num_workers"]
    vocab_size = parameter_dict["vocab_size"]
                
    
    # 生成数据集
    dataset_train = LJSpeechDataset(ark_path_train, max_length_src, max_length_tgt, corpus_path, sp_model_path)
    dataset_val = LJSpeechDataset(ark_path_val, max_length_src, max_length_tgt, corpus_path, sp_model_path)
        
    # 生成data_loader
    # 这里生成的是一个迭代器,返回值batch在第一个，需要设置lstm类参数batch_first
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    
    # for best performance
    checkpoint_best = ModelCheckpoint(save_path=best_path)
    
    evaluator = ModelEvaluator()
        
    # 实例化模型
    # print(dataset_train.word_to_index["-"])
    model = TransformerForTTS(max_length_src, max_length_tgt, vocab_size, n_feature, d_model, nhead, num_encoder_layers, num_decoder_layers)
    model = model.cuda()
    trainables = [p for p in model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} k'.format(sum(p.numel() for p in model.parameters()) / 1e3))
    print('Total trainable parameter number is : {:.3f} k'.format(sum(p.numel() for p in trainables) / 1e3))
        
    # 定义损失函数
    # 对批次损失值默认求平均，可以改成sum求和
    if loss_name == 'MSE':
        loss_fun = nn.MSELoss(reduction='mean').cuda()
    elif loss_name == 'MAE':
        loss_fun = nn.L1Loss().cuda()
    else:
        raise ValueError(f"Unsupported loss type: {loss_name}")
        
    # 定义优化器
    if optimizer_name == 'adam':
        # weight_decay(权重衰减): L2正则化    
        # betas: # 动量系数，控制一阶和二阶动量的平滑效果,默认为(0.9, 0.999), 越高越平滑，同时可能导致模型对梯度变化反应更慢
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-7, betas=(0.9, 0.999))
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
    # 用于混合精度训练（Mixed Precision Training）的工具，通常与自动混合精度（Automatic Mixed Precision, AMP）一起使用。
    scaler = GradScaler()

    # 创建ReduceLROnPlateau调度器.
    # 在patience个epoch里没有显著改善，则lr=lr*factor
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    run = Runner(model, optimizer, loss_fun, scaler)
    
    # 开始训练
    loss_train_list = []
    loss_val_list = []
    best = None
    best_eval_index_value = 0
    
    for epoch in range(n_epoch):
        # torch.cuda.empty_cache()
        # 训练
        print(f'\n\nepoch: {epoch+1}\nThe performance in the dataset_train:')
        loss_train = run(data_loader_train)
        loss_train_list.append(loss_train)
        print(f'The loss of train: {loss_train:.5f}')
        # 验证
        print('The performance in the dataset_val:')
        loss_val, prediction_all, label_all = run(data_loader_val, train=False)
        loss_val_list.append(loss_val) 
        print(f"The loss of val: {loss_val:.5f}\n")
        # 根据loss调整lr
        scheduler.step(loss_train)
        mcd, sc = evaluator.evaluate_tts(prediction_all, label_all)
        # performence = f'Performance:\nepoch: {epoch}\nWER: {wer:.4f}\nCER: {cer:.4f}\nMER: {mer:.4f}\nWIL: {wil:.4f}\n'
        # print(performence)
        # save best 
        if checkpoint_best.save_best(model, optimizer, epoch, mcd):
            save_param_dict(best_path, max_length_src, max_length_tgt, vocab_size, n_feature, d_model, nhead, num_decoder_layers, num_decoder_layers, ark_path_test, corpus_path, sp_model_path, batch_size, num_workers)
            best_eval_index_value = mcd
            best = f'best mcd:{mcd}, sc:{sc}'

    
    evaluator.draw_curve(log_folder,loss_train_list, 'loss_train')
    evaluator.draw_curve(log_folder, loss_val_list, 'loss_val')

    print(f'\n\n\nThe hyperparameter list:\n{parameter_dict}')
    print(best)
    
    return best_eval_index_value
        
    
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
            
            best_eval_index_value = main(argument, parameter_dict)
            
            # 恢复原始的sys.stdout
            sys.stdout = original_stdout
            os.chdir(f'{pwd}/log')
            new_logger_folder = f'{log_folder}_{best_eval_index_value:.4f}'
            os.rename(log_folder, new_logger_folder)
    else:
        print(f'parameter_group.json error:\n{parameter_group}')
        exit(1)
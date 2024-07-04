import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.cuda.amp import autocast
from generic import resource_monitor

from torch.autograd import Variable
import math
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x.size(1) 1为seq_Len
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    
class TransformerEncoderForScoring(nn.Module):
    def __init__(self, n_feature, d_model, n_head, max_len, num_encoder_layers=6, l2_lambda=0, use_checkpoint=True):
        super(TransformerEncoderForScoring, self).__init__()
        
        self.d_model = d_model
        self.l2_lambda = l2_lambda
        self.use_checkpoint = use_checkpoint
        
        self.change_d_model = nn.Linear(n_feature, d_model)
        
        self.position_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=n_head, num_encoder_layers=num_encoder_layers, dim_feedforward=d_model*4, batch_first=False)
        
        # 池化层
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc = nn.Linear(d_model, 1)

    @autocast()
    @resource_monitor
    def forward(self, src, src_key_padding_mask):
        src = self.change_d_model(src)
        # 添加位置编码
        src = src + self.position_encoder(src)
        # 通过 Transformer 编码器
        if self.use_checkpoint:
            transformer_output = checkpoint(self.encode, src, src_key_padding_mask)
        else:
            transformer_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # 池化
        pooled_output = self.avg_pool(transformer_output.permute(1, 2, 0)).squeeze(-1)
        # 全连接层
        output = self.fc(pooled_output)
        output = torch.sigmoid(output) * 5
        return output
    
    def encode(self, src, src_key_padding_mask):
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg
    


class TransformerForTTS(nn.Module):
    '''
    检查点技术的适应范围：
    1、模型开始部分不使用。这是因为检查点简单地通过检查输入张量的 requires_grad 行为来决定它的输入函数是否需要梯度下降(例如，它是否处于 requires_grad=True 或 requires_grad=False模式)。模型的输入张量几乎总是处于 requires_grad=False 模式，因为我们感兴趣的是计算相对于网络权重而不是输入样本本身的梯度。因此，模型中的第一个子模块应用检查点没多少意义: 它反而会冻结现有的权重，阻止它们进行任何训练。
    2、任何在重新运行时表现出非幂等(non-idempotent)行为的层。如dropout, BN
    3、必要时可以通过特殊方式使第一层拥有一个无意义的梯度。https://mathpretty.com/11156.html
    '''
    def __init__(self, n_feature, max_length_src, max_length_tgt, vocab_size, n_mel_channels=80, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(TransformerForTTS, self).__init__()
        
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_length_src)
        # 线性层将音频特征调整为transformer模型的维度
        self.change_d_model = nn.Linear(n_feature, d_model)
        self.position_decoder = PositionalEncoding(d_model, max_length_tgt)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=d_model*4, dropout=dropout, norm_first=False, batch_first=True)
        # 输出层
        self.output_linear = nn.Linear(d_model, n_mel_channels)
        # nn.init.xavier_uniform_(self.output_linear.weight)
        
        
    def warp_transform(self, model, src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask):
        return model(src=src, tgt=tgt, src_key_padding_mask =src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask)
    
    @autocast()
    def forward(self, src, src_key_padding_mask, tgt, tgt_key_padding_mask, tgt_mask):

        # src = src.transpose(0, 1)
        # tgt = tgt.transpose(0, 1)  # [seq_len_tgt, batch_size]
        # print("src shape:", src.shape)  # [16, 2451, 23]
        # print("tgt shape:", tgt.shape)  # [16, 77]
        
        src = self.encoder_embedding(src)
        # print("tgt shape:", tgt.shape)  # [bs, txt_seq_len, d_model]
        src += self.position_decoder(src)
        
        tgt = self.change_d_model(tgt)
        # src = self.bn1(src.transpose(1, 2)).transpose(1, 2)
        tgt += self.position_encoder(tgt)
        # print("src shape:", src.shape)  # [bs, mel_seq_len, d_model]
        

        # (tgt_max_length, batch_size, d_model)
        prediction = checkpoint(self.warp_transform, self.transformer, src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask)
        # print("prediction shape:", prediction.shape)    # [16, 77, 512]

        # 输出层77 32 32434
        prediction = checkpoint(self.output_linear, prediction)

        return prediction
    
    def l2_regularization(self, l2_lambda=0):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return l2_lambda * l2_reg
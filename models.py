import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.cuda.amp import autocast

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

class TransformerForTTS(nn.Module):
    '''
    检查点技术的适应范围：
    1、模型开始部分不使用。这是因为检查点简单地通过检查输入张量的 requires_grad 行为来决定它的输入函数是否需要梯度下降(例如，它是否处于 requires_grad=True 或 requires_grad=False模式)。模型的输入张量几乎总是处于 requires_grad=False 模式，因为我们感兴趣的是计算相对于网络权重而不是输入样本本身的梯度。因此，模型中的第一个子模块应用检查点没多少意义: 它反而会冻结现有的权重，阻止它们进行任何训练。
    2、任何在重新运行时表现出非幂等(non-idempotent)行为的层。如dropout, BN
    3、必要时可以通过特殊方式使第一层拥有一个无意义的梯度。https://mathpretty.com/11156.html
    '''
    def __init__(self, max_length_src, max_length_tgt, vocab_size, n_mel_channels=80, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(TransformerForTTS, self).__init__()
        
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_length_src)
        # 线性层将音频特征调整为transformer模型的维度
        self.change_d_model = nn.Linear(n_mel_channels, d_model)
        self.position_decoder = PositionalEncoding(d_model, max_length_tgt)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward=d_model*4, dropout=dropout, norm_first=False, batch_first=True)
        # 输出层
        self.output_linear = nn.Linear(d_model, n_mel_channels)
        
        self.tgt_mask = self.create_look_ahead_mask(max_length_tgt)
        # nn.init.xavier_uniform_(self.output_linear.weight)
        
        
    def warp_transform(self, model, src, tgt, src_key_padding_mask, tgt_key_padding_mask, tgt_mask):
        return model(src=src, tgt=tgt, src_key_padding_mask =src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, tgt_mask=tgt_mask)
    
    @autocast()
    def forward(self, src, src_key_padding_mask, tgt, tgt_key_padding_mask):
        
        src = self.encoder_embedding(src)
        src += self.position_encoder(src)
        
        tgt = self.change_d_model(tgt)
        tgt += self.position_decoder(tgt)
        

        # (tgt_max_length, batch_size, d_model)
        prediction = checkpoint(self.warp_transform, self.transformer, src, tgt, src_key_padding_mask, tgt_key_padding_mask, self.tgt_mask)

        # 输出层16 1008 80
        prediction = checkpoint(self.output_linear, prediction)

        return prediction
    
    def create_look_ahead_mask(self, max_len):
        """
        生成前瞻掩码。
        :param size: 文本序列长度。
        """
        # mask = (torch.triu(torch.ones((max_len, max_len))) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = (torch.triu(torch.ones((max_len, max_len)), diagonal=1) == 1)
        return mask.cuda()
    
    def l2_regularization(self, l2_lambda=0):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return l2_lambda * l2_reg
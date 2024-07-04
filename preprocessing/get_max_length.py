import torch
import kaldiio
import sentencepiece as spm

train_fbank = '/home/you/workspace/son/transformer-based-TTS/feat/train/fbank.ark'
val_fbank = '/home/you/workspace/son/transformer-based-TTS/feat/val/fbank.ark'
test_fbank = '/home/you/workspace/son/transformer-based-TTS/feat/test/fbank.ark'
def get_max_wav(ark_path):
    feats_dict = kaldiio.load_ark(ark_path)
    feats_list = [[torch.tensor(feat), uttid] for uttid, feat in feats_dict]
    length_max = 0
    for feat_label in feats_list:
        length = feat_label[0].shape[0]
        d_num = feat_label[0].shape[1]
        if length > length_max: 
            length_max = length
    return length_max

def get_max_corpus(corpus_path, sp_model_path):
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        max_len = 0
        for line in f:
            length = len(sp.encode(line))
            max_len = max(max_len, length)
    return max_len


if __name__ == '__main__':
    l_train = get_max_wav(train_fbank)
    l_val = get_max_wav(val_fbank)
    l_test = get_max_wav(test_fbank)
    print(f'max seq_length of audio is {l_train}, {l_val}, {l_test}')
# 1008 1008 1008
    max_corpus = get_max_corpus('/home/you/workspace/son/transformer-based-TTS/preprocessing/corpus.txt', '/home/you/workspace/son/transformer-based-TTS/preprocessing/sentencepiece.model')
    print(f'max seq_length of text is {max_corpus}')
# 65
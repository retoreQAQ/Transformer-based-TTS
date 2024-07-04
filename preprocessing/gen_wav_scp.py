import os
import random

def get_wav_files(wav_dir):
    wav_files = []
    for root, _, files in os.walk(wav_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def split_dataset(wav_files, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(wav_files)
    total = len(wav_files)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_files = wav_files[:train_end]
    val_files = wav_files[train_end:val_end]
    test_files = wav_files[val_end:]
    
    return train_files, val_files, test_files

# 写入 .scp 文件
def write_scp_file(file_list, scp_path):
    with open(scp_path, 'w') as f:
        for file_path in file_list:
            file_name = os.path.basename(file_path).replace('.wav', '')
            f.write(f"{file_name} {file_path}\n")
            
wav_dir = '/home/you/workspace/database/LJSpeech-1.1/wavs'
wav_files = get_wav_files(wav_dir)
train_files, val_files, test_files = split_dataset(wav_files)
write_scp_file(train_files, '/home/you/workspace/son/transformer-based-TTS/feat/train/wav.scp')
write_scp_file(val_files, '/home/you/workspace/son/transformer-based-TTS/feat/val/wav.scp')
write_scp_file(test_files, '/home/you/workspace/son/transformer-based-TTS/feat/test/wav.scp')

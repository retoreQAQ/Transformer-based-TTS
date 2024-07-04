#!/bin/bash
featbin_path="/home/you/kaldi/src/featbin"&&
root_path="/home/you/workspace/son/transformer-based-TTS"&&
feat_folder="feat"&&
sub_folder_list=("train" "val" "test")&&
cd $featbin_path&&
for sub_folder in "${sub_folder_list[@]}"; do
    work_path="$root_path/$feat_folder/$sub_folder"
    # compute-mfcc-feats --allow-downsample=true scp:$work_path/wav.scp ark:$work_path/mfcc_13.ark&&
    # ./apply-cmvn-sliding ark:$work_path/mfcc_13.ark ark:$work_path/mfcc_13_norm.ark&&
    # ./add-deltas ark:$work_path/mfcc_13_norm.ark ark:$work_path/mfcc.ark&&
    # compute-spectrogram-feats --allow-downsample=true scp:$work_path/wav.scp ark:$work_path/spec.ark&&
    compute-fbank-feats --allow-downsample=true scp:$work_path/wav.scp ark:$work_path/fbank.ark
    # compute-plp-feats --allow-downsample=true scp:$work_path/wav.scp ark:$work_path/plp_13.ark&&
    # ./apply-cmvn-sliding ark:$work_path/plp_13.ark ark:$work_path/plp_13_norm.ark&&
    # ./add-deltas ark:$work_path/plp_13_norm.ark ark:$work_path/plp.ark
done
#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME


vocab_file="/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
input_file="/search/odin/liruihong/word-based-transformer/data/train_data_1000k.tsv"
output_file="/search/odin/liruihong/word-based-transformer/data/pretrain_data/pretrain_1000k.tfrecord"

python create_pretraining_data.py \
    --vocab_file=$vocab_file \
    --input_file=$input_file \
    --output_file=$output_file \
    --max_seq_length=256 \
    --masked_lm_prob=0.1 \
    --max_predictions_per_seq=25 \
    --do_token=0

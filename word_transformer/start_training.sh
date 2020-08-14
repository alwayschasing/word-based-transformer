#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME


config_file="/search/odin/liruihong/word-based-transformer/config_data/model_config.json"
vocab_file="/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
input_file="/search/odin/liruihong/word-based-transformer/data/train_data_1000k.tsv"
cached_tfrecord="/search/odin/liruihong/word-based-transformer/cached_data/train_data_1000k.tfrecord"
output_dir="/search/odin/liruihong/word-based-transformer/model_output/word-transformer"

python keyword_extract.py \
    --gpu_id="2" \
    --config_file=$config_file \
    --vocab_file=$vocab_file \
    --output_dir=$output_dir \
    --input_file=$input_file \
    --cached_tfrecord=$cached_tfrecord \
    --embedding_table_trainable=True \
    --max_seq_length=256 \
    --save_checkpoint_steps=20000 \
    --do_train=True \
    --batch_size=64 \
    --num_warmup_steps=1000 \
    --num_train_epochs=3

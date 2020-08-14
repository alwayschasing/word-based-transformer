#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME


config_file="/search/odin/liruihong/word-based-transformer/config_data/model_config.json"
vocab_file="/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
input_file="/search/odin/liruihong/word-based-transformer/data/train_data"
cached_tfrecord=""
output_dir="/search/odin/liruihong/word-based-transformer/model_output/word-transformer"

python keyword_extract.py \
    --gpu_id="2" \
    --config_file=$config_file \
    --vocab_file=$vocab_file \
    --output_dir=$output_dir \
    --input_file=$input_file \
    --embedding_table_trainable=True \
    --embedding_size=128 \
    --max_seq_length=256 \
    --save_checkpoint_steps=20 \
    --do_train=True \
    --batch_size=32 \
    --num_warmup_steps=10 \
    --num_train_epochs=3

#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME


config_file="/search/odin/liruihong/word-based-transformer/config_data/model_config.json"
vocab_file="/search/odin/liruihong/word2vec_embedding/2000000-small.txt"
input_file=""
cached_tfrecord=""
output_dir=""

python keyword_extract.py \
    --config_file=$config_file \
    --vocab_file=$vocab_file \
    --output_dir=$output_dir \
    --input_file=""
    --embedding_table_trainable=True \
    --embedding_size=128 \
    --max_seq_length=256 \
    --save_checkpoint_steps=2000 \
    --do_train=True \
    --num_warmup_steps=1000 \
    --num_train_epochs=3

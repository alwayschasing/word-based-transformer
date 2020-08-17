#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME


config_file="/search/odin/liruihong/word-based-transformer/config_data/model_config.json"
vocab_file="/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
#input_file="/search/odin/liruihong/word-based-transformer/data/train_data_1000k.tsv"
#input_file="/search/odin/liruihong/word-based-transformer/data/dev_data_100k.tsv"
#cached_train_data="/search/odin/liruihong/word-based-transformer/cached_data/pretrain_data_1000k.tfrecord"
input_file="/search/odin/liruihong/word-based-transformer/cached_data/pretrain_data_1000k.tfrecord"
cached_dev_data="/search/odin/liruihong/word-based-transformer/cached_data/dev_data_100k.tfrecord"
output_dir="/search/odin/liruihong/word-based-transformer/model_output/pretrain_bertattn_clssify"
embedding_table="/search/odin/liruihong/word-based-transformer/config_data/final_vocab_embedding.txt"
eval_model=""

python pretraining.py \
    --gpu_id="6" \
    --model_name="bert" \
    --task_type="classify" \
    --config_file=$config_file \
    --vocab_file=$vocab_file \
    --output_dir=$output_dir \
    --input_file=$input_file \
    --cached_tfrecord=$cached_train_data \
    --embedding_table_trainable=True \
    --max_seq_length=256 \
    --mask_num=25 \
    --save_checkpoint_steps=20000 \
    --do_train=True \
    --do_eval=False \
    --batch_size=64 \
    --num_warmup_steps=1000 \
    --num_train_steps=50000

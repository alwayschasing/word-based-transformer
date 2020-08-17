#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME


config_file="/search/odin/liruihong/word-based-transformer/config_data/model_config.json"
vocab_file="/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
#input_file="/search/odin/liruihong/word-based-transformer/data/train_data_1000k.tsv"
input_file="/search/odin/liruihong/word-based-transformer/data/text_pair.tsv"
#input_file="/search/odin/liruihong/word-based-transformer/data/dev_data_100k.tsv"
cached_train_data="/search/odin/liruihong/word-based-transformer/cached_data/train_text_pair.tfrecord"
cached_dev_data="/search/odin/liruihong/word-based-transformer/cached_data/dev_data_100k.tfrecord"
output_dir="/search/odin/liruihong/word-based-transformer/model_output/loademb_bilstm_regression_test"
embedding_table="/search/odin/liruihong/word-based-transformer/config_data/final_vocab_embedding.txt"
eval_model=""

python keyword_extract.py \
    --gpu_id="4" \
    --model_name="bilstm" \
    --task_type="regression" \
    --do_token=True \
    --config_file=$config_file \
    --vocab_file=$vocab_file \
    --output_dir=$output_dir \
    --input_file=$input_file \
    --cached_data=$cached_train_data \
    --embedding_table=$embedding_table \
    --embedding_table_trainable=True \
    --max_seq_length=256 \
    --save_checkpoint_steps=20000 \
    --do_train=True \
    --do_eval=False \
    --batch_size=32 \
    --num_warmup_steps=100 \
    --num_train_epochs=5

#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME


config_file="/search/odin/liruihong/word-based-transformer/config_data/model_config.json"
vocab_file="/search/odin/liruihong/word-based-transformer/config_data/final_vocab.txt"
#vocab_file="/search/odin/liruihong/dssm/dssm-master/vocab.txt"
input_file="/search/odin/liruihong/word-based-transformer/data/cutword_article_1000k.tsv"
#input_file="/search/odin/liruihong/word-based-transformer/data/text_pair.tsv"
#input_file="/search/odin/liruihong/word-based-transformer/data/dev_data_100k.tsv"
cached_train_data="/search/odin/liruihong/word-based-transformer/cached_data/cutword_article_1000k.tfrecord"
cached_dev_data="/search/odin/liruihong/word-based-transformer/cached_data/dev_data_100k.tfrecord"
output_dir="/search/odin/liruihong/word-based-transformer/model_output/bertattn_batch512_neg50_lr0.1"
embedding_table="/search/odin/liruihong/word-based-transformer/config_data/final_vocab_embedding.txt"
init_checkpoint="/search/odin/liruihong/word-based-transformer/model_output/loademb_bilstm_regression_test/model.ckpt-0"
eval_model=""
    #--init_checkpoint=$init_checkpoint \
    #--embedding_table=$embedding_table \
    #--cached_tfrecord=$cached_train_data \

python keyword_extract.py \
    --gpu_id="6" \
    --model_name="bert" \
    --task_type="regression" \
    --do_token=False \
    --config_file=$config_file \
    --vocab_file=$vocab_file \
    --output_dir=$output_dir \
    --input_file=$input_file \
    --cached_tfrecord=$cached_train_data \
    --embedding_table_trainable=True \
    --max_seq_length=256 \
    --save_checkpoint_steps=1000 \
    --do_train=True \
    --do_eval=False \
    --batch_size=512 \
    --num_warmup_steps=1000 \
    --NEG=50 \
    --learning_rate=0.01 \
    --num_train_epochs=5

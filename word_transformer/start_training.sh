#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.0:$CUDA_HOME

<<<<<<< HEAD

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
=======
python keyword_extract.py \
    --config_file="/search/odin/liruihong/NLPTool/config_data/model_config.json" \
    --vocab_file="/search/odin/liruihong/word2vec_embedding/2000000-small.txt" \
    --stop_words_file="/search/odin/liruihong/word2vec_embedding/cn_stopwords.txt" \
    --embedding_table="/search/odin/liruihong/word2vec_embedding/2000000-small.txt" \
    --use_pos=True \
    --output_dir="/search/odin/liruihong/NLPTool/model_output/article2d_small_epoch2_test" \
    --embedding_table_trainable=False \
    --embedding_size=200 \
>>>>>>> parent of 53115a1... commit change
    --max_seq_length=256 \
    --save_checkpoint_steps=20 \
    --do_train=True \
<<<<<<< HEAD
    --batch_size=32 \
    --num_warmup_steps=10 \
    --num_train_epochs=3
=======
    --train_data="/search/odin/liruihong/NLPTool/datasets/article2d_pos.tfrecord" \
    --num_warmup_steps=1000 \
    --num_train_steps=20000
>>>>>>> parent of 53115a1... commit change

#!/usr/bin/env python
# -*- encoding:utf-8 -*-


def mergeclean_raw_vocabs(vocab_files, output_vocab_file):
    final_vocab = set()
    for file_name in vocab_files:
        with open(file_name, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split('\t')
                word = parts[0]
                if len(parts) > 1:
                    count = int(parts[1])
                    if count < 50:
                        continue
                final_vocab.add(word)

    with open(output_vocab_file, "w", encoding="utf-8") as wfp: 
        for word in final_vocab:
            wfp.write("%s\n" % (word))


if __name__ == "__main__":
    vocab_files = ["/search/odin/liruihong/word-based-transformer/data/extracted_vocab", "/search/odin/liruihong/word2vec_embedding/keywords_vocab"] 
    output_vocab_file = "/search/odin/liruihong/word-based-transformer/data/final_vocab.txt"
    mergeclean_raw_vocabs(vocab_files, output_vocab_file)
        
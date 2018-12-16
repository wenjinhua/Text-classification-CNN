from __future__ import print_function
from gensim.models import word2vec
from os.path import join, exists, split
import os
import numpy as np


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=290, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
   
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    num_features    # Word vector dimensionality                      
    min_word_count  # Minimum word count                        
    context         # Context window size 
    """
    model_dir = 'models'
    model_name = "{:d}features_{:d}minwords_{:d}context".format(num_features, min_word_count, context)#{:d}是10进制,相当于c语言的%d
    model_name = join(model_dir, model_name)
    if exists(model_name):
        embedding_model = word2vec.Word2Vec.load(model_name)
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]#sentences是将索引矩阵变换为单词的对应矩阵
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)
        #该构造函数执行了3个步骤：建立空的模型、简历词典、训练模型
        #sentences可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或·ineSentence构建。
        #workers:控制训练的并行数
        #size:输出的词向量的维度(特征数量)
        #min_count:最小出现次数
        #window 为训练的窗口大小，8表示每个词考虑前8个词与后8个词
        #sample:表示 采样的阈值，如果一个词在训练样本中出现的频率越大，那么就越会被采样 

        # If we don't plan to train the model any further, calling 
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)#锁定模型方法

        # Saving the model for later use. You can load it later using Word2Vec.load()
        if not exists(model_dir):
            os.mkdir(model_dir)
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
                              np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    print(len(embedding_weights))
    #个人觉得没必要：vocabulary_inv中是出现次数大于等于1的单词，所以embedding_model中的词都在字典中，所以没有必要加入其他词。经测试二者的长度都是18765
    #embedding_weights和embedding_model相比只是多了一些不在该模型中的词
    #uniform从一个均匀分布[low,high)中随机采样， size: 输出样本数目
    return embedding_weights


if __name__ == '__main__':
    import data_helpers #引入自己写的模块

    print("Loading data...")
    x, _, _, vocabulary_inv_list = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}#vocabulary_inv是（索引：单词）的顺序
    w = train_word2vec(x, vocabulary_inv)

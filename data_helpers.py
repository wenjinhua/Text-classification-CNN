import numpy as np
import re
import itertools
from collections import Counter
import os
import string
from gensim.models import word2vec

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
"""


def clean_str(strText):
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
                #string.punctuation：所有标点
                #dict（）：创建一个字典并返回
                #ord（）：返回字符串对应的ascii码
    no_punctuation = strText.translate(remove_punctuation_map)
    return no_punctuation.strip()

#多分类
def load_data_and_labels():
    print("loading data_and_labels...")
    x_text = []
    y = []
    data_path="D:/software/git/Git/myTest/data"
   # with open(data_path, 'r', encoding='UTF-8') as f:
    labelDirList = os.listdir(data_path)

    for labelDir in labelDirList:
        lablePath = os.path.join(data_path,labelDir)
        label = labelDir.split('_')[0]
        fileList = os.listdir(lablePath)
       
        for fileName in fileList:#fileName是txt文档
            filePath= os.path.join(lablePath,fileName)
            with open(filePath, 'r', encoding='gbk') as f:
                #获取每一行的数据
                fileText=f.read()#一次性获取txt文件里的所有内容
                if(fileText != ""):
                    fileText = clean_str(fileText)
                    fileText = fileText.split(" ")#分词后的数据
                    x_text.append(fileText)
    
                    #获取每一行数据的标签并向量化
                    one_hot=[0]*15#一共有15个类别
                    one_hot[int(label)]=1
                    y.append(np.array(one_hot))  
    return [x_text, np.array(y)]#返回去掉特殊符号的并用空格分词的文本，文本的标签

def pad_sentences(sentences, padding_word="<PAD/>"):
    #以某一行最大的长度为基准，其它行补"<PAD/>"，从而使所有行都有相同的长度
    sequence_length = max(len(x) for x in sentences)#x是每一行分词后的所有单词的最大长度
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    #chain把一组迭代对象串联起来，形成一个更大的迭代器。把所有行的数据（二维数据）串联形成一维数组
    #Counter（计数器）是对字典的补充，用于追踪值的出现次数。形成'hounds':1的格式
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]#vocabulary_inv是出现次数最多的文本列表
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    #enumerate函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列,
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()#获得所有数据和标签的list
    sentences_padded = pad_sentences(sentences)#将所有行变成相同的长度
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)#vocabulary是出现次数最多的单词和索引的dict（文本：索引），vocabulary_inv是出现最多的单词列表
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    #x是将每一行的数据根据单词在字典中出现的索引位置形成数组，如某一行为[1,35,456,0,0,...]
    return [x, y, vocabulary, vocabulary_inv]
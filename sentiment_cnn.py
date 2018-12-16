# 问题：1、训练集、验证集、测试集的选择
#     2、函数模型和序列模型，函数模型不能使用model.predict_classes
#     3、

import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding,Convolution2D,MaxPooling2D
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
np.random.seed(0)

model_type = "CNN-non-static" 
data_source = "local_dir" 

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 10

# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10


def load_data(data_source):
    assert data_source in ["keras_data_set", "local_dir"], "Unknown data source"
    if data_source == "keras_data_set":
        (x_train, y_train), (x_dev, y_dev) = imdb.load_data(num_words=max_words, start_char=None,
                                                              oov_char=None, index_from=None)

        x_train = sequence.pad_sequences(x_train, maxlen=sequence_length, padding="post", truncating="post")
        x_dev = sequence.pad_sequences(x_dev, maxlen=sequence_length, padding="post", truncating="post")

        vocabulary = imdb.get_word_index()
        vocabulary_inv = dict((v, k) for k, v in vocabulary.items())
        vocabulary_inv[0] = "<PAD/>"
    else:
        data_path="D:/software/git/Git/myTest/data"
        x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
        vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
        #y = y.argmax(axis=1)#argmax返回第1维的最大值
        # Shuffle data
        shuffle_indices = np.random.permutation(np.arange(len(y)))#permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组
        x = x[shuffle_indices]#按照shuffle_indices的顺序重新排序x
        y = y[shuffle_indices]#按照shuffle_indices的顺序重新排序y

        train_len = int(len(x) * 0.7) 
        #将数据的前70%作为训练集，15%作为验证集，15%作为测试集
        x_train = x[:train_len]
        y_train = y[:train_len]
        x_copy = x[train_len:]
        y_copy = y[train_len:]

        dev_len = int(len(x_copy) * 0.5) 
        x_dev = x_copy[:dev_len]
        y_dev = y_copy[:dev_len]
        x_test = x_copy[dev_len:]
        y_test = y_copy[dev_len:]

        print(x)
        print(x_dev)

    return x_train, y_train, x_dev, y_dev , x_test, y_test, vocabulary_inv


# Data Preparation
print("Load data...")
x_train, y_train, x_dev , y_dev , x_test, y_test, vocabulary_inv = load_data(data_source)

if sequence_length != x_dev.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_dev.shape[1]

print("x_train shape:", x_train.shape)
print("x_dev shape:", x_dev.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_dev)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    #np.vstack它是垂直（按照行顺序）的把数组给堆叠起来
    if model_type == "CNN-static":
        #stack为numpy数组堆叠的函数
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_dev = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_dev])
        print("x_train static shape:", x_train.shape)
        print("x_dev static shape:", x_dev.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")

# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)
# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)
    #input_dim字典长度,output_dim代表全连接嵌入的维度,input_length：当输入序列的长度固定时，该值为其长度。
z = Dropout(dropout_prob[0])(z)
#Dropout是在一次循环中我们先随机选择神经层中的一些单元并将其临时隐藏，然后再进行该次循环中神经网络的训练和优化过程。

# Convolutional block
conv_blocks = []
for sz in filter_sizes:#filter_sizes是过滤器的大小
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    #filters是通道的个数（特征的个数），即卷积核的数目（即输出的维度）,kernel_size为卷积核的大小，如（3,8）,padding为边界模式，为“valid”, “same” 或“full”，strides是步长，
    #权重(卷积核/过滤器)指的是数字filters，接受域指的是权重(卷积核)的大小kernel_size。
    conv = MaxPooling1D(pool_size=2)(conv)#ool_size：整数，池化窗口大小
    conv = Flatten()(conv)#把多维的输入一维化
    conv_blocks.append(conv)
  
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(15, activation="sigmoid")(z)
model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#loss是损失函数，计算预测值和真实值的偏差
#optimizer是优化器，用来更新和计算影响模型训练和模型输出的网络参数，使其逼近或达到最优值，从而最小化(或最大化)损失函数E(x)
#metrics是性能评估函数

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

#Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
        validation_data=(x_dev, y_dev),verbose=2)
#fit函数返回值是一个History对象，History类对象包含两个属性：分别为epoch和history，epoch为训练轮数。history字典类型，包含val_loss,val_acc,loss,acc四个key值。
#batch_size指定进行梯度下降时每个batch包含的样本数
#epochs训练的轮数，训练数据将会被遍历nb_epoch次
#validation_data形式为（X，y）或（X，y，sample_weights）的tuple，是指定的验证集。
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

model.save('D:/software/git/Git/myTest/model/my_model.h5')
result = model.evaluate(x_test,y_test,batch_size=32,verbose=1)
print(result)
# result = model.predict(x_test,batch_size=32, verbose=2)
# resultMax = np.argmax(result,axis=1)
# print(resultMax)#argmax的返回值是最大的数的索引。axis=1是同一个一维数组内比较
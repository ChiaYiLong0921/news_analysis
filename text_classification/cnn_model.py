# coding: utf-8

import tensorflow as tf
from tensorflow.keras import layers, Model

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 10  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(Model):
    """文本分类，CNN模型"""

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config

        self.embedding = layers.Embedding(config.vocab_size, config.embedding_dim)
        self.conv = layers.Conv1D(config.num_filters, config.kernel_size, activation='relu')
        self.gmp = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(config.hidden_dim, activation='relu')
        self.dropout = layers.Dropout(1 - config.dropout_keep_prob) # Keras dropout uses drop rate
        self.fc2 = layers.Dense(config.num_classes, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv(x)
        x = self.gmp(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)



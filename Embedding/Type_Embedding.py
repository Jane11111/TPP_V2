# -*- coding: utf-8 -*-
# @Time    : 2020-08-18 20:16
# @Author  : zxl
# @FileName: Category_Embedding.py

import numpy as np
import tensorflow as tf
from Embedding.base_embedding import Base_embedding


class Type_embedding(Base_embedding):

    def __init__(self,is_training = True, type_num = 0, max_seq_len = 0, sims_len = 0, FLAGS = None):
        super(Type_embedding, self).__init__(is_training= is_training)

        self.type_num = type_num
        self.max_seq_len = max_seq_len
        self.sims_len = sims_len
        self.FLAGS = FLAGS

    def init_placeholders(self):

        with tf.variable_scope("input_layer"):
            self.type_lst = tf.placeholder(tf.int32, [None,None], name = 'type_lst')
            self.time_lst = tf.placeholder(tf.float32, [None,None], name = 'time_lst')
            # self.position_lst = tf.placeholder(tf.int32, [None, self.max_seq_len], name = 'position_lst')

            self.target_type = tf.placeholder(tf.int32, [None], name = 'target_type')
            self.target_time = tf.placeholder(tf.float32, [None], name = 'target_time')
            # 当前这个sequence的长度
            self.seq_len = tf.placeholder(tf.int32,[None,], name= 'seq_len')
            # # 当前sequence采样的长度
            # self.sims_len = tf.placeholder(tf.int32, [None,], name = 'sims_len')
            # 当前sequence采样的时间lst 已补齐
            self.sims_time_lst = tf.placeholder(tf.float32, [None, None], name= 'sims_time_lst')

    def get_type_embedding(self,type_lst):
        """

        :param type_lst: N,
        :return: N, type_emb_size
        """

        type_embedding = tf.nn.embedding_lookup(self.type_emb_lookup_table,type_lst)


        return type_embedding


    def get_embedding(self, num_units):

        self.type_emb_lookup_table = self.init_embedding_lookup_table(name = 'type',
                                                                      total_count = self.type_num,
                                                                      embedding_dim = num_units,
                                                                      is_training = self.is_training)

        type_lst_embedding = tf.nn.embedding_lookup(self.type_emb_lookup_table,self.type_lst)


        target_type_embedding = tf.nn.embedding_lookup(self.type_emb_lookup_table, self.target_type)


        return type_lst_embedding,\
               self.time_lst,\
               target_type_embedding,\
               self.target_type,\
               self.target_time,\
               self.seq_len,\
               self.sims_time_lst

    def make_feed_dic(self, batch_data):

        feed_dic = {}
        type_lst = []
        time_lst = []
        target_type= []
        target_time = []
        seq_len = []
        sims_time_lst = []

        for example in batch_data:
            padding_size = [0,self.max_seq_len - example[4]]

            type_lst.append(np.pad(example[0], padding_size, 'constant'))
            time_lst.append(np.pad(example[1], padding_size, 'constant'))

            target_type.append(example[2])
            target_time.append(example[3])
            seq_len.append(example[4])
            sims_time_lst.append(example[5])
        feed_dic[self.type_lst] = np.array(type_lst)
        feed_dic[self.time_lst] = np.array(time_lst)
        feed_dic[self.target_type] = np.array(target_type)
        feed_dic[self.target_time] = np.array(target_time)
        feed_dic[self.seq_len] = np.array(seq_len)
        feed_dic[self.sims_time_lst] = np.array(sims_time_lst)


        return feed_dic

























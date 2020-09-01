# -*- coding: utf-8 -*-
# @Time    : 2020-08-18 20:16
# @Author  : zxl
# @FileName: Category_Embedding.py

import numpy as np
import tensorflow as tf
from Embedding.base_embedding import Base_embedding


class history_embedding(Base_embedding):

    def __init__(self,is_training = True, type_num = 0, max_seq_len = 0, sims_len = 0, FLAGS = None):
        super(history_embedding, self).__init__(is_training= is_training)

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
            self.T_lst = tf.placeholder(tf.float32, [None, ], name='T_lst')
            self.not_first_lst = tf.placeholder(tf.float32, [None, ], name='not_first_lst')
            # # 当前sequence采样的长度
            # self.sims_len = tf.placeholder(tf.int32, [None,], name = 'sims_len')
            # 当前sequence采样的时间lst 已补齐
            self.sims_time_lst = tf.placeholder(tf.float32, [None, None], name= 'sims_time_lst')
            self.target_time_last_lst = tf.placeholder(tf.float32, [None,None], name = 'target_time_last_lst')
            self.target_time_now_lst = tf.placeholder(tf.float32, [None, None], name='target_time_now_lst')
            self.sim_time_last_lst = tf.placeholder(tf.float32, [None,None, None], name='sim_time_last_lst')
            self.sim_time_now_lst = tf.placeholder(tf.float32, [None,None, None], name='sim_time_now_lst')




    def get_type_embedding(self,type_lst):
        """

        :param type_lst: N,
        :return: N, type_emb_size
        """

        type_embedding = tf.nn.embedding_lookup(self.type_emb_lookup_table,type_lst)


        return type_embedding


    def get_embedding(self, num_units):

        self.type_emb_lookup_table = self.init_embedding_lookup_table(name = 'type',
                                                                      total_count = self.type_num + 3,
                                                                      embedding_dim = num_units,
                                                                      is_training = self.is_training)

        type_lst_embedding = tf.nn.embedding_lookup(self.type_emb_lookup_table,self.type_lst)


        target_type_embedding = tf.nn.embedding_lookup(self.type_emb_lookup_table, self.target_type)

        # time embedding for THP
        M = num_units
        single_odd_mask = np.zeros(shape=(M,))
        single_odd_mask[::2] = 1
        single_odd_mask = tf.convert_to_tensor(single_odd_mask, dtype=tf.float32)  # M,
        single_even_mask = np.zeros(shape=(M,))
        single_even_mask[1::2] = 1
        single_even_mask = tf.convert_to_tensor(single_even_mask, dtype=tf.float32)

        emb_time_lst = tf.tile(tf.expand_dims(self.time_lst, axis=2), [1, 1, M])  # batch_size, seq_len, M

        single_odd_deno = tf.to_float(10000 ** (tf.range(start=0, limit=M, delta=1) / M))  # M,
        single_even_deno = tf.to_float(10000 ** (tf.range(start=1, limit=M + 1, delta=1) / M))

        odd_emb = tf.cos(emb_time_lst / single_odd_deno)
        even_emb = tf.sin(emb_time_lst / single_even_deno)
        self.time_lst_emb = odd_emb * single_odd_mask + even_emb * single_even_mask

        return type_lst_embedding,\
               self.time_lst,\
               self.time_lst_emb,\
               target_type_embedding,\
               self.target_type,\
               self.target_time,\
               self.seq_len,\
               self.T_lst,\
               self.not_first_lst,\
               self.sims_time_lst,\
               self.target_time_last_lst,\
               self.target_time_now_lst,\
               self.sim_time_last_lst,\
               self.sim_time_now_lst

    def make_feed_dic(self, batch_data):
        feed_dic = {}
        type_lst = []
        time_lst = []
        target_type= []
        target_time = []
        seq_len = []
        T_lst = []
        not_first_lst = []
        sims_time_lst = []
        target_time_last_lst =[]
        target_time_now_lst = []
        sim_time_last_lst = []
        sim_time_now_lst =[]


        for example in batch_data:
            padding_size = [0,self.max_seq_len - example[4]]

            type_lst.append(np.pad(example[0], padding_size, 'constant'))
            time_lst.append(np.pad(example[1], padding_size, 'constant'))

            target_type.append(example[2])
            target_time.append(example[3])
            seq_len.append(example[4])
            T_lst.append(example[5])
            not_first_lst.append(example[6])
            sims_time_lst.append(example[7])
            target_time_last_lst.append(np.pad(example[8][0],padding_size,'constant'))
            target_time_now_lst.append(np.pad(example[8][1],padding_size,'constant'))
            tmp_sim_last = []
            tmp_sim_now = []
            for sims_time_last_now_list in example[9]:
                tmp_sim_last.append(np.pad(sims_time_last_now_list[0], padding_size, 'constant'))
                tmp_sim_now.append(np.pad(sims_time_last_now_list[1], padding_size, 'constant'))

            sim_time_last_lst.append(tmp_sim_last)
            sim_time_now_lst.append(tmp_sim_now)
        feed_dic[self.type_lst] = np.array(type_lst)
        feed_dic[self.time_lst] = np.array(time_lst)
        feed_dic[self.target_type] = np.array(target_type)
        feed_dic[self.target_time] = np.array(target_time)
        feed_dic[self.seq_len] = np.array(seq_len)
        feed_dic[self.T_lst] = np.array(T_lst)
        feed_dic[self.not_first_lst] = np.array(not_first_lst)
        feed_dic[self.sims_time_lst] = np.array(sims_time_lst)

        feed_dic[self.target_time_last_lst]=np.array(target_time_last_lst)
        feed_dic[self.target_time_now_lst]=np.array(target_time_now_lst)
        feed_dic[self.sim_time_last_lst]=np.array(sim_time_last_lst) # batch_size, sim_len, num_units
        feed_dic[self.sim_time_now_lst] = np.array(sim_time_now_lst)


        return feed_dic

























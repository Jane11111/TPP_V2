# -*- coding: utf-8 -*-
# @Time    : 2020/9/13 9:53
# @Author  : zxl
# @FileName: HP.py


"""
hawkes process
"""

import tensorflow as tf
from Model.base_model import base_model
from Model.Modules.transformer_encoder import transformer_encoder
from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.Modules.time_prediction import hp_time_predictor,ihp_time_predictor
from Model.Modules.type_prediction import thp_type_predictor
from Model.Modules.continuous_time_rnn import ContinuousLSTM
from tensorflow.python.ops import array_ops
from Model.Modules.intensity_calculation import hp_intensity_calculation,ihp_intensity_calculation

class HP_model(base_model):
    def __init__(self, FLAGS, Embedding, sess):

        super(HP_model,self).__init__(FLAGS= FLAGS,
                                                Embedding= Embedding)
        self.now_batch_size = tf.placeholder(tf.int32, shape = [], name = 'bath_size')

        self.type_emb_size = self.FLAGS.type_emb_size
        self.num_units = self.FLAGS.num_units # attention的num
        self.num_heads = self.FLAGS.num_heads
        self.num_blocks = self.FLAGS.num_blocks
        self.dropout_rate = self.FLAGS.dropout
        self.regulation_rate = self.FLAGS.regulation_rate
        self.type_num = self.FLAGS.type_num
        self.sims_len = self.FLAGS.sims_len
        self.max_seq_len = self.FLAGS.max_seq_len

        self.type_lst, \
        self.type_lst_embedding, \
        self.time_lst, \
        self.time_lst_embedding, \
        self.sahp_time_lst_embedding, \
        self.target_type_embedding, \
        self.target_type,\
        self.target_time, \
        self.seq_len, \
        self.T_lst,\
        self.not_first_lst,\
        self.sims_time_lst,\
        self.target_time_last_lst,\
        self.target_time_now_lst, \
        self.sim_time_last_lst, \
        self.sim_time_now_lst = self.embedding.get_embedding(self.type_emb_size)
        self.mask_index = tf.reshape(self.seq_len - 1, [-1, 1])

        self.build_model()
        self.init_variables(sess)

class HP(HP_model):


    def build_model(self):



        with tf.variable_scope('intensity_calculation', reuse=tf.AUTO_REUSE):
            intensity_model = hp_intensity_calculation(type_num=self.type_num,dtype=self.time_lst.dtype)

            self.target_lambda = intensity_model.cal_target_intensity(timenow_lst=self.target_time_now_lst,
                                                                      type_lst=self.type_lst,
                                                                      type_num = self.type_num,
                                                                      seq_len = self.seq_len,
                                                                      max_seq_len = self.max_seq_len
                                                                      )
            self.sims_lambda = intensity_model.cal_sims_intensity(sims_timenow_lst = self.sim_time_now_lst,
                                                                  type_lst=self.type_lst,
                                                                  type_num=self.type_num,
                                                                  seq_len=self.seq_len,
                                                                  max_seq_len=self.max_seq_len,
                                                                  sims_len=self.sims_len)

        with tf.variable_scope('type_time_calculation', reuse=tf.AUTO_REUSE):

            last_time = tf.squeeze(gather_indexes(batch_size=self.now_batch_size,
                                                  seq_length=self.max_seq_len,
                                                  width=1,
                                                  sequence_tensor=tf.expand_dims(self.time_lst, axis=-1),
                                                  positions=self.mask_index - 1))  # target_time 上个时刻
            time_predictor = hp_time_predictor(f = intensity_model,
                                               type_num = self.type_num, max_seq_len = self.max_seq_len,
                                               type_lst = self.type_lst, last_time = last_time,time_lst=self.time_lst,
                                               seq_len = self.seq_len)
            self.predict_time = time_predictor.predict_time(outer_sims_len=self.FLAGS.outer_sims_len)


            self.predict_type_prob = self.target_lambda# batch_size, type_num


        self.output()


class IHP(HP_model):


    def build_model(self):



        with tf.variable_scope('intensity_calculation', reuse=tf.AUTO_REUSE):
            intensity_model = ihp_intensity_calculation(type_num=self.type_num,dtype=self.time_lst.dtype)

            self.target_lambda = intensity_model.cal_target_intensity(timenow_lst=self.target_time_now_lst,
                                                                      type_lst=self.type_lst,
                                                                      type_num = self.type_num,
                                                                      seq_len = self.seq_len,
                                                                      max_seq_len = self.max_seq_len
                                                                      )
            self.sims_lambda = intensity_model.cal_sims_intensity(sims_timenow_lst = self.sim_time_now_lst,
                                                                  type_lst=self.type_lst,
                                                                  type_num=self.type_num,
                                                                  seq_len=self.seq_len,
                                                                  max_seq_len=self.max_seq_len,
                                                                  sims_len=self.sims_len)

        with tf.variable_scope('type_time_calculation', reuse=tf.AUTO_REUSE):

            last_time = tf.squeeze(gather_indexes(batch_size=self.now_batch_size,
                                                  seq_length=self.max_seq_len,
                                                  width=1,
                                                  sequence_tensor=tf.expand_dims(self.time_lst, axis=-1),
                                                  positions=self.mask_index - 1))  # target_time 上个时刻
            time_predictor = ihp_time_predictor(f = intensity_model,type_num = self.type_num,
                                                max_seq_len = self.max_seq_len,
                                                type_lst = self.type_lst, last_time = last_time,
                                                time_lst=self.time_lst,seq_len = self.seq_len)
            self.predict_time = time_predictor.predict_time(inner_sims_len=self.FLAGS.inner_sims_len,
                                                            outer_sims_len=self.FLAGS.outer_sims_len)


            self.predict_type_prob = self.target_lambda# batch_size, type_num


        self.output()



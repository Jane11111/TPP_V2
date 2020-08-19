# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 09:51
# @Author  : zxl
# @FileName: AttentionTPP.py


import tensorflow as tf
from Model.base_model import base_model
from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.Modules.intensity_calculation import mlt_intensity


class AttentionTPP_model(base_model):

    def __init__(self, FLAGS, Embedding, sess):

        super(AttentionTPP_model,self).__init__(FLAGS= FLAGS,
                                                Embedding= Embedding)
        self.now_batch_size = tf.placeholder(tf.int32, shape = [], name = 'bath_size')

        self.type_emb_units = self.FLAGS.type_emb_units
        self.num_units = self.FLAGS.num_units # attention的num
        self.num_heads = self.FLAGS.num_heads
        self.num_blocks = self.FLAGS.num_blocks
        self.dropout_rate = self.FLAGS.dropout_rate
        self.regulation_rate = self.FLAGS.regulation_rate
        self.type_num = self.FLAGS.type_num
        self.sims_len = self.FLAGS.sims_len
        self.max_seq_len = self.FLAGS.max_seq_len

        self.type_lst_embedding, \
        self.time_lst, \
        self.target_type_embedding, \
        self.target_type,\
        self.target_time, \
        self.seq_len, \
        self.sims_time_lst = self.embedding.get_embedding(self.type_emb_units)


        self.build_model()
        self.init_variables(sess)


class AttentionTPP_MLT(AttentionTPP_model):


    def generate_emb(self,type_lst_emb, time_lst, target_type, target_time, seq_len, sims_lst):
        """

        :param type_lst_emb: batch_size, max_seq_len, type_emb_units
        :param time_lst:  batch_size, max_seq_len
        :param target_type: batch_size,
        :param target_time: batch_size,
        :param seq_len: batch_size,
        :param sims_lst: batch_size, sims_len
        :return: batch_size, 1 + sims_len, num_units 经过attention后得到的embedding
        """

        target_time = tf.reshape(target_time,[-1,1]) # batch_size, 1
        target_sims_time = tf.concat([target_time,sims_lst], axis = 2) # batch_size, 1+ sims_len

        target_type = tf.reshape(target_type, [-1,1])
        target_type_emb = self.embedding.get_type_embedding(target_type) # batch_size, 1, type_emb_units
        target_sims_type_emb = tf.tile(target_type_emb, [1,self.sims_len + 1, 1])

        query_len = tf.ones_like(seq_len, dtype = tf.int32) * (self.sims_len + 1)

        with tf.variable_scope('target_decoder', reuse = tf.AUTO_REUSE):
            time_aware_attention = Time_Aware_Attention()

            next_emb = time_aware_attention.vanilla_attention(enc = type_lst_emb,
                                                               dec = target_sims_type_emb,
                                                               num_units = self.num_units,
                                                               num_heads=self.num_heads,
                                                               num_blocks=self.num_blocks,
                                                               dropout_rate=self.dropout_rate,
                                                               is_training=True,
                                                               reuse = True,# TODO 这里写true还是false
                                                               key_length=seq_len,
                                                               query_length = query_len,
                                                               t_querys=target_sims_time,
                                                               t_keys = time_lst,
                                                               t_keys_length=self.max_seq_len,
                                                               t_querys_length = self.sims_len + 1)
            next_emb = layer_norm(next_emb)
        predict_emb = tf.reshape(next_emb,[-1,self.sims_len + 1, self.num_units]) # batch_size, 1+sims_len, num_units
        return predict_emb



    def build_model(self):

        with tf.variable_scope('intensity_calculation', reuse = tf.AUTO_REUSE):

            intensity_fun = mlt_intensity()

            layer_units = [16]

            target_demo = tf.reshape(self.seq_len,[-1,1]) # batch_size, 1
            sims_demo = tf.reshape(tf.tile(self.seq_len, [self.sims_len]) ,[-1,self.sims_len,1])# batch_size , sims_len,1


            predict_target_intensity = tf.zeros_like(target_demo) # batch_size, 1
            predict_sims_intensity = tf.zeros_like(sims_demo) # batch_size, sims_len , 1

            for type in range(self.type_num):
                target_type = tf.ones_like(self.seq_len) * type # batch_size,

                predict_emb = self.generate_emb(type_lst_emb= self.type_lst_embedding,
                                                time_lst = self.time_lst,
                                                target_type=target_type,
                                                target_time=self.target_time,
                                                sims_lst=self.sims_time_lst,
                                                seq_len = self.seq_len) # batch_size, 1+sims_len, num_units

                predict_emb = tf.reshape(predict_emb, [-1, self.num_units]) # batch_size * (1+sims_len), num_units

                predict_type_intensity = intensity_fun.cal_type_intensity(emb = predict_emb,
                                                                          type = type,
                                                                          layer_units= layer_units)
                predict_type_intensity = tf.reshape(predict_type_intensity,[-1,1+self.sims_len, 1])

                predict_target_type_intensity = predict_type_intensity[:,0,:] # batch_size, 1
                predict_sims_type_intensity =predict_type_intensity[:,1:,:]
                predict_sims_type_intensity = tf.reshape(predict_sims_type_intensity,[-1,1]) # batch_size * sims_len , 1

                predict_target_intensity = tf.concat([predict_target_intensity, predict_target_type_intensity], axis=1)
                predict_sims_intensity = tf.concat([predict_sims_intensity, predict_sims_type_intensity], axis = 1)


                self.lambda_prob = predict_target_intensity[:,1:] # batch_size, type_num
        self.output()




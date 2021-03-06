# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 09:51
# @Author  : zxl
# @FileName: AttentionTPP.py

import tensorflow.compat.v1 as tf
from Model.Modules.gru import GRU
from Model.base_model import base_model
from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.Modules.type_aware_attention import Type_Aware_Attention
from Model.Modules.intensity_calculation import mlt_intensity,single_intensity,e_intensity,type_intensity,w_intensity
from Model.Modules.type_prediction import e_type_predictor
from Model.Modules.time_prediction import thp_time_predictor
from Model.Modules.transformer_encoder import transformer_encoder
from Model.Modules.multihead_attention import Attention

class AttentionTPP_model(base_model):

    def __init__(self, FLAGS, Embedding, sess):

        super(AttentionTPP_model,self).__init__(FLAGS= FLAGS,
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


"""
------------以下是生成模型------------
"""


# class MTAM_TPP_wendy_att_time_new(AttentionTPP_model):
#
#     def get_emb(self, target_time, time_last, time_now):
#         with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):
#             time_aware_gru_net_input = tf.concat([self.type_lst_embedding,
#                                                   tf.expand_dims(time_last, 2),
#                                                   tf.expand_dims(time_now, 2)],
#                                                  axis=2)
#             self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
#                                                                         input_data=time_aware_gru_net_input,
#                                                                         input_length=tf.add(self.seq_len, -1),  #
#                                                                         type='new',
#                                                                         scope='gru')
#             short_term_intent = gather_indexes(batch_size=self.now_batch_size,
#                                                seq_length=self.max_seq_len,
#                                                width=self.num_units,
#                                                sequence_tensor=self.short_term_intent_temp,
#                                                positions=self.mask_index - 1)  # batch_size, num_units
#             self.interval_bar = short_term_intent[:, -1]  # 最后一位是interval batch_size,
#             short_term_intent = short_term_intent[:, :-1]  # 前面是h
#             short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
#             short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
#         with tf.variable_scope('long-term', reuse=tf.AUTO_REUSE):
#             predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc=self.type_lst_embedding,
#                                                                              dec=short_term_intent4vallina,
#                                                                              num_units=self.num_units,
#                                                                              num_heads=self.num_heads,
#                                                                              num_blocks=self.num_blocks,
#                                                                              dropout_rate=self.dropout_rate,
#                                                                              is_training=True,
#                                                                              reuse=False,
#                                                                              key_length=self.seq_len,  # TODO 这里为什么不减去1
#                                                                              query_length=tf.ones_like(
#                                                                                  short_term_intent4vallina[:, 0, 0],
#                                                                                  dtype=tf.int32),
#                                                                              t_querys=tf.expand_dims(target_time, 1),
#                                                                              t_keys=self.time_lst,
#                                                                              t_keys_length=self.max_seq_len,
#                                                                              t_querys_length=1
#                                                                              )
#             predict_lambda_emb = tf.reshape(predict_lambda_emb, [-1, self.num_units])
#
#
#         return layer_norm(predict_lambda_emb), self.interval_bar
#
#     def predict_time(self,time_last,interval_bar):
#         with tf.variable_scope('predict_time', reuse=tf.AUTO_REUSE):
#             self.last_time = gather_indexes(batch_size=self.now_batch_size,
#                                             seq_length=self.max_seq_len,
#                                             width=1,
#                                             sequence_tensor=self.time_lst,
#                                             positions=self.mask_index - 1)  # 上一个时间
#             emb4vallina = tf.expand_dims(self.target_type_embedding, axis=1)
#             self.interval = self.time_aware_attention.vanilla_attention4time(enc=self.type_lst_embedding,
#                                                                              dec=emb4vallina,
#                                                                              num_units=self.num_units,
#                                                                              num_heads=self.num_heads,
#                                                                              num_blocks=self.num_blocks,
#                                                                              dropout_rate=self.dropout_rate,
#                                                                              is_training=True,
#                                                                              reuse=False,
#                                                                              key_length=self.seq_len - 1,  # TODO 需要减1
#                                                                              query_length=tf.ones_like(
#                                                                                  emb4vallina[:, 0, 0],
#                                                                                  dtype=tf.int32),
#                                                                              timelast_lst=time_last,
#                                                                              last_time=self.last_time,
#                                                                              time_interval=tf.expand_dims(
#                                                                                  interval_bar, axis=1),
#                                                                              t_keys=self.time_lst,
#                                                                              t_keys_length=self.max_seq_len,
#                                                                              t_querys_length=1)
#
#         return self.last_time + self.interval
#
#     def build_model(self):
#         self.time_aware_attention = Time_Aware_Attention()
#         self.attention = Attention()
#         self.gru_net_ins = GRU()
#         self.transformer_model = transformer_encoder()
#
#         predict_target_lambda_emb, interval_bar = self.get_emb(self.target_time, self.target_time_last_lst,
#                                                                self.target_time_now_lst)
#         predict_time = self.predict_time(self.target_time_last_lst,interval_bar)
#
#         # sims_time_lst: batch_size, sims_len
#         predict_sims_emb = tf.zeros([self.now_batch_size, 1])
#         self.test = tf.split(self.sims_time_lst, self.sims_len, 1)
#         sims_time = tf.squeeze(tf.split(self.sims_time_lst, self.sims_len, 1), 2)
#         sims_time_last = tf.squeeze(tf.split(self.sim_time_last_lst, self.sims_len, 1), 2)
#         sims_time_now = tf.squeeze(tf.split(self.sim_time_now_lst, self.sims_len, 1), 2)
#         for i in range(self.sims_len):
#             # 第i个时间 batch_size, num_units
#             cur_sims_emb, _ = self.get_emb(sims_time[i], sims_time_last[i], sims_time_now[i])
#             predict_sims_emb = tf.concat([predict_sims_emb, cur_sims_emb], axis=1)
#
#         predict_sims_emb = predict_sims_emb[:, 1:]  # batch_size, sims_len * num_units
#         predict_sims_emb = tf.reshape(predict_sims_emb,
#                                       [-1, self.sims_len, self.num_units])  # batch_size, sims_len , num_units
#
#         self.predict_target_emb = predict_target_lambda_emb  #
#         self.predict_sims_emb = predict_sims_emb
#         self.emb_for_time = predict_time
#
#         with tf.variable_scope('prepare_emb'):
#             emb_for_intensity = self.predict_target_emb
#             emb_for_time = self.emb_for_time
#             emb_for_type = self.predict_target_emb
#
#         with tf.variable_scope('intensity_calculation'):
#             type_lookup_table = self.embedding.type_emb_lookup_table[:-3, :]  # type_num   , num_units
#             intensity_model = e_intensity(W=type_lookup_table, type_num=self.type_num)
#
#             self.target_lambda = intensity_model.cal_target_intensity(emb_for_intensity)
#             self.sims_lambda = intensity_model.cal_sims_intensity(self.predict_sims_emb, max_sims_len=self.sims_len,
#                                                                   num_units=self.num_units)
#
#         with tf.variable_scope('type_time_calculation'):
#             time_predictor = thp_time_predictor()
#             self.predict_time = predict_time
#
#             type_predictor = e_type_predictor()  # 用E W(emb)
#             self.predict_type_prob = type_predictor.predict_type(emb=emb_for_type,
#                                                                  num_units=self.num_units,
#                                                                  type_num=self.type_num,
#                                                                  type_table=self.embedding.type_emb_lookup_table)
#             # self.predict_type_prob = tf.matmul(self.predict_type_prob, self.embedding.type_emb_lookup_table[:-3, :],
#             #                                    transpose_b=True)
#
#         self.output()


class TANHP_v3(AttentionTPP_model):
    """

    强度和type完全共享
    """
    def get_rnn_emb(self,time_last,time_now):
        with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):

            time_aware_gru_net_input = tf.concat([ self.type_lst_embedding ,
                                              tf.expand_dims(time_last, 2),
                                              tf.expand_dims(time_now, 2)],
                                             axis=2)
            self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_len,-1),  #
                                                                          type='new',
                                                                          scope='gru')
            short_term_intent = gather_indexes(batch_size=self.now_batch_size,
                                           seq_length=self.max_seq_len,
                                           width=self.num_units,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units
            interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:-1]# 前面是h
            short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
            short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
        return short_term_intent4vallina,interval_bar

    def get_type_emb(self,short_term_intent4vallina,type_lst_embedding,seq_len,time_lst,target_time):
        """
        用rnn embedding 做query
        用target time
        :param short_term_intent4vallina:
        :param target_time:
        :return:
        """
        with tf.variable_scope('type_long-term', reuse=tf.AUTO_REUSE):
            predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc= type_lst_embedding,
                                                                             dec=short_term_intent4vallina,
                                                                             num_units=self.num_units,
                                                                             num_heads=self.num_heads,
                                                                             num_blocks=self.num_blocks,
                                                                             dropout_rate=self.dropout_rate,
                                                                             is_training=True,
                                                                             reuse=False,
                                                                             key_length= seq_len,  # TODO 这里为什么不减去1
                                                                             query_length=tf.ones_like(
                                                                                 short_term_intent4vallina[:, 0, 0],
                                                                                 dtype=tf.int32),
                                                                             t_querys=tf.expand_dims(target_time, 1),
                                                                             t_keys= time_lst,
                                                                             t_keys_length=self.max_seq_len,
                                                                             t_querys_length=1
                                                                             )
            predict_lambda_emb = tf.reshape(predict_lambda_emb, [-1, self.num_units])
            predict_lambda_emb = layer_norm(predict_lambda_emb)

        return predict_lambda_emb


    def get_intensity_emb(self,target_type_embedding,type_lst_embedding,seq_len,time_lst,target_time ):
        """
        用target type做query 计算intensity embedding
        以及 target time
        :param target_type_embedding:
        :param target_time:
        :return:
        """
        target_type_embedding = tf.reshape(target_type_embedding,[-1,1,self.num_units])
        with tf.variable_scope('intensity_long-term', reuse=tf.AUTO_REUSE):
            predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc= type_lst_embedding,
                                                                             dec=target_type_embedding,
                                                                             num_units=self.num_units,
                                                                             num_heads=self.num_heads,
                                                                             num_blocks=self.num_blocks,
                                                                             dropout_rate=self.dropout_rate,
                                                                             is_training=True,
                                                                             reuse=False,
                                                                             key_length= seq_len,  # TODO 这里为什么不减去1
                                                                             query_length=tf.ones_like(
                                                                                 target_type_embedding[:, 0, 0],
                                                                                 dtype=tf.int32),
                                                                             t_querys=tf.expand_dims(target_time, 1),
                                                                             t_keys= time_lst,
                                                                             t_keys_length=self.max_seq_len,
                                                                             t_querys_length=1
                                                                             )
            predict_lambda_emb = tf.reshape(predict_lambda_emb, [-1, self.num_units])
            predict_lambda_emb = layer_norm(predict_lambda_emb)

        return predict_lambda_emb



    def predict_time(self,time_last, short_term_intent4vallina,interval_bar):

        with tf.variable_scope('predict_time',reuse= tf.AUTO_REUSE):
            self.last_time = gather_indexes(batch_size=self.now_batch_size,
                                               seq_length=self.max_seq_len,
                                               width=1,
                                               sequence_tensor=self.time_lst,
                                               positions=self.mask_index - 1)# 上一个时间
            self.interval = self.time_aware_attention.vanilla_attention4time(enc=self.type_lst_embedding,
                                                                 dec=short_term_intent4vallina,
                                                                 num_units=self.num_units,
                                                                 num_heads=self.num_heads,
                                                                 num_blocks=self.num_blocks,
                                                                 dropout_rate=self.dropout_rate,
                                                                 is_training=True,
                                                                 reuse=False,
                                                                 key_length=self.seq_len-1, # TODO 需要减1
                                                                 query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
                                                                                           dtype=tf.int32),
                                                                 timelast_lst = time_last,
                                                                 last_time = self.last_time,
                                                                 time_interval=tf.expand_dims(interval_bar,axis=1),
                                                                 t_keys=self.time_lst,
                                                                 t_keys_length=self.max_seq_len,
                                                                 t_querys_length=1)
        return self.last_time+self.interval


    def build_model(self):
        self.time_aware_attention = Time_Aware_Attention()
        self.attention = Attention()
        self.gru_net_ins = GRU()
        self.transformer_model = transformer_encoder()

        type_predictor = e_type_predictor()  # 用E W(emb)
        intensity_predictor = w_intensity(self.num_units)

        with tf.variable_scope('time_aware_rnn'):
            short_term_intent4vallina, interval_bar = self.get_rnn_emb(self.target_time_last_lst,self.target_time_now_lst) # batch_size, ...

        with tf.variable_scope('time_att'):
            self.predict_time = self.predict_time(self.target_time_last_lst,short_term_intent4vallina,interval_bar)

        with tf.variable_scope('type_att'):
            predict_target_type_emb = self.get_type_emb(short_term_intent4vallina=short_term_intent4vallina,
                                                        type_lst_embedding=self.type_lst_embedding,
                                                        seq_len = self.seq_len,
                                                        time_lst = self.time_lst,
                                                        target_time = self.target_time) # batch_size, num_units

            self.predict_type_prob = type_predictor.predict_type(emb=predict_target_type_emb,
                                                                 num_units=self.num_units,
                                                                 type_num=self.type_num,
                                                                 type_table=self.embedding.type_emb_lookup_table)

        with tf.variable_scope('intensity_att'):
            predict_target_intensity_emb = self.get_intensity_emb(target_type_embedding= self.target_type_embedding,
                                                        type_lst_embedding=self.type_lst_embedding,
                                                        seq_len=self.seq_len,
                                                        time_lst=self.time_lst,
                                                        target_time=self.target_time)  # batch_size, num_units

            predict_target_inensity = intensity_predictor.cal_target_intensity(predict_target_intensity_emb) # batch_size, 1


            type_lst_embedding_for_sims = tf.tile(self.type_lst_embedding, [1, self.sims_len, 1])
            type_lst_embedding_for_sims = tf.reshape(type_lst_embedding_for_sims,
                                                     [-1, self.max_seq_len, self.num_units])
            seq_len_for_sims = tf.reshape(self.seq_len, (-1, 1))
            seq_len_for_sims = tf.reshape(tf.tile(seq_len_for_sims, [1, self.sims_len]),
                                          (-1,))  # batch_size * sims_len,
            time_lst_for_sims = tf.tile(self.time_lst, [1, self.sims_len])  # batch_size, max_seq_len * sims_len
            time_lst_for_sims = tf.reshape(time_lst_for_sims, [-1, self.max_seq_len])
            target_time_for_sims = tf.reshape(self.sims_time_lst, shape=(-1,))  # batch_size * sims_len,

            predict_sims_intensity = [] #
            tmp = tf.to_int32(tf.ones_like(target_time_for_sims))# batch_size * sims_len,
            for i in range(self.type_num):
                sims_type_embedding = self.embedding.get_type_embedding(tmp*i)
                emb = self.get_intensity_emb(target_type_embedding = sims_type_embedding,
                                             type_lst_embedding=type_lst_embedding_for_sims,
                                             seq_len = seq_len_for_sims,
                                             time_lst = time_lst_for_sims,
                                             target_time = target_time_for_sims)
                cur_intensity = intensity_predictor.cal_target_intensity(emb)
                predict_sims_intensity.append(cur_intensity)
            predict_sims_intensity = tf.concat(predict_sims_intensity,axis=1) # batch_size * sims_len, type_num
        self.sims_lambda = tf.reshape(predict_sims_intensity,[-1,self.sims_len,self.type_num])
        self.target_lambda = tf.tile(predict_target_inensity,[1,self.type_num])
        #
        # tmp = tf.to_float(tf.reshape(tf.ones_like(self.seq_len),[-1,1]))
        #
        # self.target_lambda = tf.tile(tmp,[1,self.type_num])
        # self.sims_lambda = tf.reshape(tf.tile(tmp,[1,self.sims_len * self.type_num]),[-1,self.sims_len, self.type_num])




        self.output()


class TANHP_v2(AttentionTPP_model):
    """

    强度和type完全共享
    """

    def get_target_emb (self,target_time,time_last,time_now ):
        with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):

            time_aware_gru_net_input = tf.concat([ self.type_lst_embedding ,
                                              tf.expand_dims(time_last, 2),
                                              tf.expand_dims(time_now, 2)],
                                             axis=2)
            self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_len,-1),  #
                                                                          type='new',
                                                                          scope='gru')
            short_term_intent = gather_indexes(batch_size=self.now_batch_size,
                                           seq_length=self.max_seq_len,
                                           width=self.num_units,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units
            self.interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:-1]# 前面是h
            short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
            short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
        with tf.variable_scope('long-term',reuse=tf.AUTO_REUSE):
            predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc=self.type_lst_embedding,
                                                                 dec=short_term_intent4vallina,
                                                                 num_units=self.num_units,
                                                                 num_heads=self.num_heads,
                                                                 num_blocks=self.num_blocks,
                                                                 dropout_rate=self.dropout_rate,
                                                                 is_training=True,
                                                                 reuse=False,
                                                                 key_length=self.seq_len, # TODO 这里为什么不减去1
                                                                 query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
                                                                                           dtype=tf.int32),
                                                                 t_querys=tf.expand_dims(target_time,1),
                                                                 t_keys=self.time_lst,
                                                                 t_keys_length=self.max_seq_len,
                                                                 t_querys_length=1
                                                                 )
            predict_lambda_emb = tf.reshape(predict_lambda_emb,[-1,self.num_units])



        return layer_norm(predict_lambda_emb), short_term_intent4vallina,self.interval_bar

    def get_sims_emb (self,target_time,time_last,time_now ):
        """

        :param target_time: batch_size * sims_len,
        :param time_last: batch_size * sims_len, max_seq_len
        :param time_now: batch_size * sims_len, max_seq_len
        :return:
        """
        type_lst_embedding = tf.tile(self.type_lst_embedding, [1,self.sims_len, 1])
        type_lst_embedding = tf.reshape(type_lst_embedding,[-1,self.max_seq_len,self.num_units])
        seq_len = tf.reshape(self.seq_len, (-1,1))
        seq_len = tf.reshape(tf.tile(seq_len, [1,self.sims_len]),(-1,)) # batch_size * sims_len,
        mask_index = tf.reshape( seq_len - 1, (-1, 1))

        cur_batch_size = self.now_batch_size*self.sims_len

        time_lst = tf.tile(self.time_lst, [1,self.sims_len]) # batch_size, max_seq_len * sims_len
        time_lst = tf.reshape(time_lst, [-1, self.max_seq_len])

        with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):


            time_aware_gru_net_input = tf.concat([ type_lst_embedding ,
                                              tf.expand_dims(time_last, 2),
                                              tf.expand_dims(time_now, 2)],
                                             axis=2)
            self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(seq_len,-1),  #
                                                                          type='new',
                                                                          scope='gru')
            short_term_intent = gather_indexes(batch_size=cur_batch_size,
                                           seq_length=self.max_seq_len,
                                           width=self.num_units,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions= mask_index -1 )#batch_size, num_units
            self.interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:-1]# 前面是h
            short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
            short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
        with tf.variable_scope('long-term',reuse=tf.AUTO_REUSE):
            predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc= type_lst_embedding,
                                                                 dec=short_term_intent4vallina,
                                                                 num_units=self.num_units,
                                                                 num_heads=self.num_heads,
                                                                 num_blocks=self.num_blocks,
                                                                 dropout_rate=self.dropout_rate,
                                                                 is_training=True,
                                                                 reuse=False,
                                                                 key_length= seq_len, # TODO 这里为什么不减去1
                                                                 query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
                                                                                           dtype=tf.int32),
                                                                 t_querys=tf.expand_dims(target_time,1),
                                                                 t_keys= time_lst,
                                                                 t_keys_length=self.max_seq_len,
                                                                 t_querys_length=1
                                                                 )
            predict_lambda_emb = tf.reshape(predict_lambda_emb,[-1,self.num_units])



        return layer_norm(predict_lambda_emb)

    def predict_time(self,time_last, short_term_intent4vallina,interval_bar):

        with tf.variable_scope('predict_time',reuse= tf.AUTO_REUSE):
            self.last_time = gather_indexes(batch_size=self.now_batch_size,
                                               seq_length=self.max_seq_len,
                                               width=1,
                                               sequence_tensor=self.time_lst,
                                               positions=self.mask_index - 1)# 上一个时间
            self.interval = self.time_aware_attention.vanilla_attention4time(enc=self.type_lst_embedding,
                                                                 dec=short_term_intent4vallina,
                                                                 num_units=self.num_units,
                                                                 num_heads=self.num_heads,
                                                                 num_blocks=self.num_blocks,
                                                                 dropout_rate=self.dropout_rate,
                                                                 is_training=True,
                                                                 reuse=False,
                                                                 key_length=self.seq_len-1, # TODO 需要减1
                                                                 query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
                                                                                           dtype=tf.int32),
                                                                 timelast_lst = time_last,
                                                                 last_time = self.last_time,
                                                                 time_interval=tf.expand_dims(interval_bar,axis=1),
                                                                 t_keys=self.time_lst,
                                                                 t_keys_length=self.max_seq_len,
                                                                 t_querys_length=1)
        return self.last_time+self.interval


    def build_model(self):
        self.time_aware_attention = Time_Aware_Attention()
        self.attention = Attention()
        self.gru_net_ins = GRU()
        self.transformer_model = transformer_encoder()

        predict_target_lambda_emb, short_term_intent4vallina, interval_bar = self.get_target_emb(self.target_time, self.target_time_last_lst,
                                                                       self.target_time_now_lst)
        predict_time = self.predict_time(time_last = self.target_time_last_lst,
                                         short_term_intent4vallina=short_term_intent4vallina,
                                         interval_bar=interval_bar)



        sims_time = tf.reshape(self.sims_time_lst,shape=(-1,)) # batch_size * sims_len,
        sims_time_last = tf.reshape(self.sim_time_last_lst, shape=(-1,self.max_seq_len)) # batch_size*sims_len, max_seq_len
        sims_time_now = tf.reshape(self.sim_time_now_lst, shape=(-1,self.max_seq_len))
        predict_sims_emb = self.get_sims_emb(sims_time,sims_time_last,sims_time_now)


        predict_sims_emb = tf.reshape(predict_sims_emb,[-1,self.sims_len,self.num_units])# batch_size, sims_len , num_units

        self.predict_target_emb = predict_target_lambda_emb #
        self.predict_sims_emb = predict_sims_emb
        self.emb_for_time = predict_time

        with tf.variable_scope('prepare_emb'):
            emb_for_intensity = self.predict_target_emb
            emb_for_time = self.emb_for_time
            emb_for_type = self.predict_target_emb

        with tf.variable_scope('type_time_calculation'):
            time_predictor = thp_time_predictor()
            self.predict_time = predict_time

            type_predictor = e_type_predictor() # 用E W(emb)
            self.predict_type_prob = type_predictor.predict_type(emb=emb_for_type,
                                                                 num_units=self.num_units,
                                                                 type_num=self.type_num,
                                                                 type_table = self.embedding.type_emb_lookup_table)
            # self.predict_type_prob = tf.matmul(self.predict_type_prob, self.embedding.type_emb_lookup_table[:-3, :],
            #                                    transpose_b=True)

        # with tf.variable_scope('intensity_calculation'):
            type_lookup_table = self.embedding.type_emb_lookup_table[:-3, :]  # type_num   , num_units
            intensity_model = type_intensity(type_predictor=type_predictor,num_units = self.num_units, W=type_lookup_table, type_num=self.type_num)

            self.target_lambda = tf.nn.softplus(intensity_model.cal_target_intensity(emb_for_intensity))
            self.sims_lambda = tf.nn.softplus(intensity_model.cal_sims_intensity(self.predict_sims_emb, max_sims_len=self.sims_len,
                                                                     num_units=self.num_units))



        self.output()

class MTAM_TPP_wendy_att_time(AttentionTPP_model):

    def get_target_emb (self,target_time,time_last,time_now ):
        with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):

            time_aware_gru_net_input = tf.concat([ self.type_lst_embedding ,
                                              tf.expand_dims(time_last, 2),
                                              tf.expand_dims(time_now, 2)],
                                             axis=2)
            self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_len,-1),  #
                                                                          type='new',
                                                                          scope='gru')
            short_term_intent = gather_indexes(batch_size=self.now_batch_size,
                                           seq_length=self.max_seq_len,
                                           width=self.num_units,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units
            self.interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:-1]# 前面是h
            short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
            short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
        with tf.variable_scope('long-term',reuse=tf.AUTO_REUSE):
            predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc=self.type_lst_embedding,
                                                                 dec=short_term_intent4vallina,
                                                                 num_units=self.num_units,
                                                                 num_heads=self.num_heads,
                                                                 num_blocks=self.num_blocks,
                                                                 dropout_rate=self.dropout_rate,
                                                                 is_training=True,
                                                                 reuse=False,
                                                                 key_length=self.seq_len, # TODO 这里为什么不减去1
                                                                 query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
                                                                                           dtype=tf.int32),
                                                                 t_querys=tf.expand_dims(target_time,1),
                                                                 t_keys=self.time_lst,
                                                                 t_keys_length=self.max_seq_len,
                                                                 t_querys_length=1
                                                                 )
            predict_lambda_emb = tf.reshape(predict_lambda_emb,[-1,self.num_units])



        return layer_norm(predict_lambda_emb), short_term_intent4vallina,self.interval_bar

    def get_sims_emb (self,target_time,time_last,time_now ):
        """

        :param target_time: batch_size * sims_len,
        :param time_last: batch_size * sims_len, max_seq_len
        :param time_now: batch_size * sims_len, max_seq_len
        :return:
        """
        type_lst_embedding = tf.tile(self.type_lst_embedding, [1,self.sims_len, 1])
        type_lst_embedding = tf.reshape(type_lst_embedding,[-1,self.max_seq_len,self.num_units])
        seq_len = tf.reshape(self.seq_len, (-1,1))
        seq_len = tf.reshape(tf.tile(seq_len, [1,self.sims_len]),(-1,)) # batch_size * sims_len,
        mask_index = tf.reshape( seq_len - 1, (-1, 1))

        cur_batch_size = self.now_batch_size*self.sims_len

        time_lst = tf.tile(self.time_lst, [1,self.sims_len]) # batch_size, max_seq_len * sims_len
        time_lst = tf.reshape(time_lst, [-1, self.max_seq_len])

        with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):


            time_aware_gru_net_input = tf.concat([ type_lst_embedding ,
                                              tf.expand_dims(time_last, 2),
                                              tf.expand_dims(time_now, 2)],
                                             axis=2)
            self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(seq_len,-1),  #
                                                                          type='new',
                                                                          scope='gru')
            short_term_intent = gather_indexes(batch_size=cur_batch_size,
                                           seq_length=self.max_seq_len,
                                           width=self.num_units,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions= mask_index -1 )#batch_size, num_units
            self.interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:-1]# 前面是h
            short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
            short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
        with tf.variable_scope('long-term',reuse=tf.AUTO_REUSE):
            predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc= type_lst_embedding,
                                                                 dec=short_term_intent4vallina,
                                                                 num_units=self.num_units,
                                                                 num_heads=self.num_heads,
                                                                 num_blocks=self.num_blocks,
                                                                 dropout_rate=self.dropout_rate,
                                                                 is_training=True,
                                                                 reuse=False,
                                                                 key_length= seq_len, # TODO 这里为什么不减去1
                                                                 query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
                                                                                           dtype=tf.int32),
                                                                 t_querys=tf.expand_dims(target_time,1),
                                                                 t_keys= time_lst,
                                                                 t_keys_length=self.max_seq_len,
                                                                 t_querys_length=1
                                                                 )
            predict_lambda_emb = tf.reshape(predict_lambda_emb,[-1,self.num_units])



        return layer_norm(predict_lambda_emb)

    def predict_time(self,time_last, short_term_intent4vallina,interval_bar):

        with tf.variable_scope('predict_time',reuse= tf.AUTO_REUSE):
            self.last_time = gather_indexes(batch_size=self.now_batch_size,
                                               seq_length=self.max_seq_len,
                                               width=1,
                                               sequence_tensor=self.time_lst,
                                               positions=self.mask_index - 1)# 上一个时间
            self.interval = self.time_aware_attention.vanilla_attention4time(enc=self.type_lst_embedding,
                                                                 dec=short_term_intent4vallina,
                                                                 num_units=self.num_units,
                                                                 num_heads=self.num_heads,
                                                                 num_blocks=self.num_blocks,
                                                                 dropout_rate=self.dropout_rate,
                                                                 is_training=True,
                                                                 reuse=False,
                                                                 key_length=self.seq_len-1, # TODO 需要减1
                                                                 query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
                                                                                           dtype=tf.int32),
                                                                 timelast_lst = time_last,
                                                                 last_time = self.last_time,
                                                                 time_interval=tf.expand_dims(interval_bar,axis=1),
                                                                 t_keys=self.time_lst,
                                                                 t_keys_length=self.max_seq_len,
                                                                 t_querys_length=1)
        return self.last_time+self.interval


    def build_model(self):
        self.time_aware_attention = Time_Aware_Attention()
        self.attention = Attention()
        self.gru_net_ins = GRU()
        self.transformer_model = transformer_encoder()

        predict_target_lambda_emb, short_term_intent4vallina, interval_bar = self.get_target_emb(self.target_time, self.target_time_last_lst,
                                                                       self.target_time_now_lst)
        predict_time = self.predict_time(time_last = self.target_time_last_lst,
                                         short_term_intent4vallina=short_term_intent4vallina,
                                         interval_bar=interval_bar)

        # sims_time_lst: batch_size, sims_len
        # predict_sims_emb = tf.zeros([self.now_batch_size, 1])
        # self.test = tf.split(self.sims_time_lst, self.sims_len, 1)
        # sims_time = tf.squeeze(tf.split(self.sims_time_lst, self.sims_len, 1), 2)
        # sims_time_last = tf.squeeze(tf.split(self.sim_time_last_lst, self.sims_len, 1), 2)
        # sims_time_now = tf.squeeze(tf.split(self.sim_time_now_lst, self.sims_len, 1), 2)
        # for i in range(self.sims_len):
        #     #第i个时间 batch_size, num_units
        #     cur_sims_emb, _ ,_= self.get_emb(sims_time[i],sims_time_last[i],sims_time_now[i])
        #     predict_sims_emb = tf.concat([predict_sims_emb,cur_sims_emb],axis = 1)
        #
        # predict_sims_emb = predict_sims_emb[:,1:] # batch_size, sims_len * num_units


        sims_time = tf.reshape(self.sims_time_lst,shape=(-1,)) # batch_size * sims_len,
        sims_time_last = tf.reshape(self.sim_time_last_lst, shape=(-1,self.max_seq_len)) # batch_size*sims_len, max_seq_len
        sims_time_now = tf.reshape(self.sim_time_now_lst, shape=(-1,self.max_seq_len))
        predict_sims_emb = self.get_sims_emb(sims_time,sims_time_last,sims_time_now)


        predict_sims_emb = tf.reshape(predict_sims_emb,[-1,self.sims_len,self.num_units])# batch_size, sims_len , num_units

        self.predict_target_emb = predict_target_lambda_emb #
        self.predict_sims_emb = predict_sims_emb
        self.emb_for_time = predict_time

        with tf.variable_scope('prepare_emb'):
            emb_for_intensity = self.predict_target_emb
            emb_for_time = self.emb_for_time
            emb_for_type = self.predict_target_emb

        with tf.variable_scope('type_time_calculation'):
            time_predictor = thp_time_predictor()
            self.predict_time = predict_time

            type_predictor = e_type_predictor() # 用E W(emb)
            self.predict_type_prob = type_predictor.predict_type(emb=emb_for_type,
                                                                 num_units=self.num_units,
                                                                 type_num=self.type_num,
                                                                 type_table = self.embedding.type_emb_lookup_table)
            # self.predict_type_prob = tf.matmul(self.predict_type_prob, self.embedding.type_emb_lookup_table[:-3, :],
            #                                    transpose_b=True)

        # with tf.variable_scope('intensity_calculation'):
            type_lookup_table = self.embedding.type_emb_lookup_table[:-3, :]  # type_num   , num_units
            intensity_model = e_intensity(type_predictor=type_predictor,num_units = self.num_units, W=type_lookup_table, type_num=self.type_num)

            self.target_lambda = intensity_model.cal_target_intensity(emb_for_intensity)
            self.sims_lambda = intensity_model.cal_sims_intensity(self.predict_sims_emb, max_sims_len=self.sims_len,
                                                                     num_units=self.num_units)



        self.output()

class TA_GRU(AttentionTPP_model):

    def get_target_emb (self,target_time,time_last,time_now ):
        with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):

            time_aware_gru_net_input = tf.concat([ self.type_lst_embedding ,
                                              tf.expand_dims(time_last, 2),
                                              tf.expand_dims(time_now, 2)],
                                             axis=2)
            self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(self.seq_len,-1),  #
                                                                          type='new',
                                                                          scope='gru')
            short_term_intent = gather_indexes(batch_size=self.now_batch_size,
                                           seq_length=self.max_seq_len,
                                           width=self.num_units,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions=self.mask_index -1 )#batch_size, num_units
            self.interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:-1]# 前面是h
            short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
            short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)

            predict_lambda_emb = short_term_intent


        return  predict_lambda_emb , short_term_intent4vallina,self.interval_bar

    def get_sims_emb (self,target_time,time_last,time_now ):
        """

        :param target_time: batch_size * sims_len,
        :param time_last: batch_size * sims_len, max_seq_len
        :param time_now: batch_size * sims_len, max_seq_len
        :return:
        """
        type_lst_embedding = tf.tile(self.type_lst_embedding, [1,self.sims_len, 1])
        type_lst_embedding = tf.reshape(type_lst_embedding,[-1,self.max_seq_len,self.num_units])
        seq_len = tf.reshape(self.seq_len, (-1,1))
        seq_len = tf.reshape(tf.tile(seq_len, [1,self.sims_len]),(-1,)) # batch_size * sims_len,
        mask_index = tf.reshape( seq_len - 1, (-1, 1))

        cur_batch_size = self.now_batch_size*self.sims_len

        time_lst = tf.tile(self.time_lst, [1,self.sims_len]) # batch_size, max_seq_len * sims_len
        time_lst = tf.reshape(time_lst, [-1, self.max_seq_len])

        with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):


            time_aware_gru_net_input = tf.concat([ type_lst_embedding ,
                                              tf.expand_dims(time_last, 2),
                                              tf.expand_dims(time_now, 2)],
                                             axis=2)
            self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
                                                                          input_data=time_aware_gru_net_input,
                                                                          input_length=tf.add(seq_len,-1),  #
                                                                          type='new',
                                                                          scope='gru')
            short_term_intent = gather_indexes(batch_size=cur_batch_size,
                                           seq_length=self.max_seq_len,
                                           width=self.num_units,
                                           sequence_tensor=self.short_term_intent_temp,
                                           positions= mask_index -1 )#batch_size, num_units
            self.interval_bar = short_term_intent[:,-1] # 最后一位是interval batch_size,
            short_term_intent = short_term_intent[:,:-1]# 前面是h
            short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
            short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
        with tf.variable_scope('long-term',reuse=tf.AUTO_REUSE):
            predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc= type_lst_embedding,
                                                                 dec=short_term_intent4vallina,
                                                                 num_units=self.num_units,
                                                                 num_heads=self.num_heads,
                                                                 num_blocks=self.num_blocks,
                                                                 dropout_rate=self.dropout_rate,
                                                                 is_training=True,
                                                                 reuse=False,
                                                                 key_length= seq_len, # TODO 这里为什么不减去1
                                                                 query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
                                                                                           dtype=tf.int32),
                                                                 t_querys=tf.expand_dims(target_time,1),
                                                                 t_keys= time_lst,
                                                                 t_keys_length=self.max_seq_len,
                                                                 t_querys_length=1
                                                                 )
            predict_lambda_emb = tf.reshape(predict_lambda_emb,[-1,self.num_units])



        return layer_norm(predict_lambda_emb)

    def predict_time(self,time_last, short_term_intent4vallina,interval_bar):

        with tf.variable_scope('predict_time',reuse= tf.AUTO_REUSE):
            self.last_time = gather_indexes(batch_size=self.now_batch_size,
                                               seq_length=self.max_seq_len,
                                               width=1,
                                               sequence_tensor=self.time_lst,
                                               positions=self.mask_index - 1)# 上一个时间
            self.interval = tf.reshape(interval_bar,[-1,1])


        return self.last_time+self.interval


    def build_model(self):
        self.time_aware_attention = Time_Aware_Attention()
        self.attention = Attention()
        self.gru_net_ins = GRU()
        self.transformer_model = transformer_encoder()

        predict_target_lambda_emb, short_term_intent4vallina, interval_bar = self.get_target_emb(self.target_time, self.target_time_last_lst,
                                                                       self.target_time_now_lst)
        predict_time = self.predict_time(time_last = self.target_time_last_lst,
                                         short_term_intent4vallina=short_term_intent4vallina,
                                         interval_bar=interval_bar)


        sims_time = tf.reshape(self.sims_time_lst,shape=(-1,)) # batch_size * sims_len,
        sims_time_last = tf.reshape(self.sim_time_last_lst, shape=(-1,self.max_seq_len)) # batch_size*sims_len, max_seq_len
        sims_time_now = tf.reshape(self.sim_time_now_lst, shape=(-1,self.max_seq_len))
        predict_sims_emb = self.get_sims_emb(sims_time,sims_time_last,sims_time_now)


        predict_sims_emb = tf.reshape(predict_sims_emb,[-1,self.sims_len,self.num_units])# batch_size, sims_len , num_units

        self.predict_target_emb = predict_target_lambda_emb #
        self.predict_sims_emb = predict_sims_emb
        self.emb_for_time = predict_time

        with tf.variable_scope('prepare_emb'):
            emb_for_intensity = self.predict_target_emb
            emb_for_time = self.emb_for_time
            emb_for_type = self.predict_target_emb

        with tf.variable_scope('type_time_calculation'):
            time_predictor = thp_time_predictor()
            self.predict_time = predict_time

            type_predictor = e_type_predictor() # 用E W(emb)
            self.predict_type_prob = type_predictor.predict_type(emb=emb_for_type,
                                                                 num_units=self.num_units,
                                                                 type_num=self.type_num,
                                                                 type_table = self.embedding.type_emb_lookup_table)
            # self.predict_type_prob = tf.matmul(self.predict_type_prob, self.embedding.type_emb_lookup_table[:-3, :],
            #                                    transpose_b=True)

        # with tf.variable_scope('intensity_calculation'):
            type_lookup_table = self.embedding.type_emb_lookup_table[:-3, :]  # type_num   , num_units
            intensity_model = e_intensity(type_predictor=type_predictor,num_units = self.num_units, W=type_lookup_table, type_num=self.type_num)

            self.target_lambda = intensity_model.cal_target_intensity(emb_for_intensity)
            self.sims_lambda = intensity_model.cal_sims_intensity(self.predict_sims_emb, max_sims_len=self.sims_len,
                                                                     num_units=self.num_units)



        self.output()


# class MTAM_TPP_wendy_time(AttentionTPP_model):
#
#     def get_emb (self,target_time,time_last,time_now):
#         with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):
#
#             time_aware_gru_net_input = tf.concat([self.type_lst_embedding ,
#                                               tf.expand_dims(time_last, 2),
#                                               tf.expand_dims(time_now, 2)],
#                                              axis=2)
#             self.short_term_intent_temp = self.gru_net_ins.time_gru_net(hidden_units=self.num_units,
#                                                                           input_data=time_aware_gru_net_input,
#                                                                           input_length=tf.add(self.seq_len,-1),  #
#                                                                           type='new',
#                                                                           scope='gru')
#             short_term_intent = gather_indexes(batch_size=self.now_batch_size,
#                                            seq_length=self.max_seq_len,
#                                            width=self.num_units,
#                                            sequence_tensor=self.short_term_intent_temp,
#                                            positions=self.mask_index -1 )#batch_size, num_units
#             interval = short_term_intent[:,-1] # 最后一位是interval batch_size,
#             short_term_intent = short_term_intent[:,:-1]# 前面是h
#             short_term_intent = tf.layers.dense(short_term_intent, self.num_units)
#             short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
#         with tf.variable_scope('long-term',reuse=tf.AUTO_REUSE):
#             predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc=self.type_lst_embedding,
#                                                                  dec=short_term_intent4vallina,
#                                                                  num_units=self.num_units,
#                                                                  num_heads=self.num_heads,
#                                                                  num_blocks=self.num_blocks,
#                                                                  dropout_rate=self.dropout_rate,
#                                                                  is_training=True,
#                                                                  reuse=False,
#                                                                  key_length=self.seq_len, # TODO 这里为什么不减去1
#                                                                  query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
#                                                                                            dtype=tf.int32),
#                                                                  t_querys=tf.expand_dims(target_time,1),
#                                                                  t_keys=self.time_lst,
#                                                                  t_keys_length=self.max_seq_len,
#                                                                  t_querys_length=1
#                                                                  )
#             predict_lambda_emb = tf.reshape(predict_lambda_emb,[-1,self.num_units])
#         with tf.variable_scope('emb_for_time',reuse= tf.AUTO_REUSE):
#             self.last_time = gather_indexes(batch_size=self.now_batch_size,
#                                                seq_length=self.max_seq_len,
#                                                width=1,
#                                                sequence_tensor=self.time_lst,
#                                                positions=self.mask_index - 1)# 上一个时间
#             self.interval = tf.reshape(interval,[-1,1])# batch_size, 1
#
#
#         return layer_norm(predict_lambda_emb), self.last_time+self.interval
#
#
#     def build_model(self):
#         self.time_aware_attention = Time_Aware_Attention()
#         self.attention = Attention()
#         self.gru_net_ins = GRU()
#         self.transformer_model = transformer_encoder()
#
#         predict_target_lambda_emb, predict_time = self.get_emb(self.target_time, self.target_time_last_lst,
#                                                                        self.target_time_now_lst)
#
#         # sims_time_lst: batch_size, sims_len
#         predict_sims_emb = tf.zeros([self.now_batch_size, 1])
#         self.test = tf.split(self.sims_time_lst, self.sims_len, 1)
#         sims_time = tf.squeeze(tf.split(self.sims_time_lst, self.sims_len, 1), 2)
#         sims_time_last = tf.squeeze(tf.split(self.sim_time_last_lst, self.sims_len, 1), 2)
#         sims_time_now = tf.squeeze(tf.split(self.sim_time_now_lst, self.sims_len, 1), 2)
#         for i in range(self.sims_len):
#             #第i个时间 batch_size, num_units
#             cur_sims_emb, _ = self.get_emb(sims_time[i],sims_time_last[i],sims_time_now[i])
#             predict_sims_emb = tf.concat([predict_sims_emb,cur_sims_emb],axis = 1)
#
#         predict_sims_emb = predict_sims_emb[:,1:] # batch_size, sims_len * num_units
#         predict_sims_emb = tf.reshape(predict_sims_emb,[-1,self.sims_len,self.num_units])# batch_size, sims_len , num_units
#
#         self.predict_target_emb = predict_target_lambda_emb #
#         self.predict_sims_emb = predict_sims_emb
#         self.emb_for_time = predict_time
#
#         with tf.variable_scope('prepare_emb'):
#             emb_for_intensity = self.predict_target_emb
#             emb_for_time = self.emb_for_time
#             emb_for_type = self.predict_target_emb
#
#
#         with tf.variable_scope('intensity_calculation'):
#             type_lookup_table = self.embedding.type_emb_lookup_table[:-3, :]  # type_num   , num_units
#             intensity_model = e_intensity(W=type_lookup_table, type_num=self.type_num)
#
#             self.target_lambda = intensity_model.cal_target_intensity(emb_for_intensity)
#             self.sims_lambda = intensity_model.cal_sims_intensity(self.predict_sims_emb, max_sims_len=self.sims_len,
#                                                                      num_units=self.num_units)
#
#         with tf.variable_scope('type_time_calculation'):
#             time_predictor = thp_time_predictor()
#             self.predict_time = predict_time
#
#             type_predictor = e_type_predictor() # 用E W(emb)
#             self.predict_type_prob = type_predictor.predict_type(emb=emb_for_type,
#                                                                  num_units=self.num_units,
#                                                                  type_num=self.type_num,
#                                                                  type_table = self.embedding.type_emb_lookup_table)
#             # self.predict_type_prob = tf.matmul(self.predict_type_prob, self.embedding.type_emb_lookup_table[:-3, :],
#             #                                    transpose_b=True)
#
#         self.output()

#
# class MTAM_only_time_aware_RNN(AttentionTPP_model,):
#     def get_emb(self,timelast_lst,timenow_lst):
#         with tf.variable_scope('short_term_intent_encoder', reuse=tf.AUTO_REUSE):
#             self.time_aware_gru_net_input = tf.concat([self.type_lst_embedding,
#                                                        tf.expand_dims(timelast_lst,2),
#                                                        tf.expand_dims(timenow_lst,2)],2)
#             self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units = self.num_units,
#                                                                               input_data=self.time_aware_gru_net_input,
#                                                                               input_length=tf.add(self.seq_len,-1),
#                                                                               type='T-SeqRec')
#             self.short_term_intent = gather_indexes(batch_size = self.now_batch_size,
#                                                     seq_length=self.max_seq_len,
#                                                     width = self.num_units,
#                                                     sequence_tensor=self.short_term_intent_temp,
#                                                     positions = self.mask_index - 1)
#             self.predict_behavior_emb = layer_norm(self.short_term_intent)
#         return self.predict_behavior_emb
#
#
#     def build_model(self):
#
#         self.gru_net_ins = GRU()
#         predict_target_lambda_emb = self.get_emb( self.target_time_last_lst, self.target_time_now_lst)
#         predict_target_lambda_emb = tf.reshape(predict_target_lambda_emb, [self.now_batch_size, self.num_units])
#
#         # sims_time_lst: batch_size, sims_len
#         predict_sims_emb = tf.zeros([self.now_batch_size, 1])
#
#         sims_time = tf.squeeze(tf.split(self.sims_time_lst, self.sims_len, 1), 2)
#         sims_time_last = tf.squeeze(tf.split(self.sim_time_last_lst, self.sims_len, 1), 2)
#         sims_time_now = tf.squeeze(tf.split(self.sim_time_now_lst, self.sims_len, 1), 2)
#         for i in range(self.sims_len):
#             # 第i个时间 batch_size, num_units
#             cur_sims_emb = self.get_emb( sims_time_last[i], sims_time_now[i])
#             cur_sims_emb = tf.reshape(cur_sims_emb, [self.now_batch_size, self.num_units])
#             predict_sims_emb = tf.concat([predict_sims_emb, cur_sims_emb], axis=1)
#
#         predict_sims_emb = predict_sims_emb[:, 1:]  # batch_size, sims_len * num_units
#         predict_sims_emb = tf.reshape(predict_sims_emb,
#                                       [-1, self.sims_len, self.num_units])  # batch_size, sims_len , num_units
#
#         self.predict_target_emb = predict_target_lambda_emb  # batch_size, num_units
#         self.predict_sims_emb = predict_sims_emb
#
#         with tf.variable_scope('intensity_calculation'):
#             type_lookup_table = self.embedding.type_emb_lookup_table[:-3,:] # type_num   , num_units
#             intensity_model = e_intensity(W = type_lookup_table, type_num=self.type_num)
#
#             self.target_intensity = intensity_model.cal_target_intensity(self.predict_target_emb)
#             self.sims_intensity = intensity_model.cal_sims_intensity(self.predict_sims_emb,max_sims_len=self.sims_len,
#                                                                      num_units=self.num_units)
#
#
#
#
#         self.output()
#
#
#
# class Vallina_Gru(AttentionTPP_model,):
#     def get_emb(self):
#         with tf.variable_scope('short_term_intent_encoder',reuse=tf.AUTO_REUSE):
#
#             self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units = self.num_units,
#                                                                   input_data=self.type_lst_embedding,
#                                                                   input_length=tf.add(self.seq_len,-1))
#             self.short_term_intent = gather_indexes(batch_size = self.now_batch_size,
#                                                     seq_length=self.max_seq_len,
#                                                     width = self.num_units,
#                                                     sequence_tensor=self.short_term_intent_temp,
#                                                     positions = self.mask_index - 1)
#             self.predict_behavior_emb = layer_norm(self.short_term_intent)
#         return self.predict_behavior_emb
#
#
#     def build_model(self):
#
#         self.gru_net_ins = GRU()
#         predict_target_lambda_emb = self.get_emb( )
#         predict_target_lambda_emb = tf.reshape(predict_target_lambda_emb, [self.now_batch_size, self.num_units])
#
#         # sims_time_lst: batch_size, sims_len
#         predict_sims_emb = tf.zeros([self.now_batch_size, 1])
#
#         sims_time = tf.squeeze(tf.split(self.sims_time_lst, self.sims_len, 1), 2)
#         sims_time_last = tf.squeeze(tf.split(self.sim_time_last_lst, self.sims_len, 1), 2)
#         sims_time_now = tf.squeeze(tf.split(self.sim_time_now_lst, self.sims_len, 1), 2)
#         for i in range(self.sims_len):
#             # 第i个时间 batch_size, num_units
#             cur_sims_emb = self.get_emb( )
#             cur_sims_emb = tf.reshape(cur_sims_emb, [self.now_batch_size, self.num_units])
#             predict_sims_emb = tf.concat([predict_sims_emb, cur_sims_emb], axis=1)
#
#         predict_sims_emb = predict_sims_emb[:, 1:]  # batch_size, sims_len * num_units
#         predict_sims_emb = tf.reshape(predict_sims_emb,
#                                       [-1, self.sims_len, self.num_units])  # batch_size, sims_len , num_units
#
#         self.predict_target_emb = predict_target_lambda_emb  # sim_len, batch_size, num_units
#         self.predict_sims_emb = predict_sims_emb
#
#         with tf.variable_scope('intensity_calculation'):
#             type_lookup_table = self.embedding.type_emb_lookup_table[:-3, :]  # type_num   , num_units
#             intensity_model = e_intensity(W=type_lookup_table, type_num=self.type_num)
#
#             self.target_intensity = intensity_model.cal_target_intensity(self.predict_target_emb)
#             self.sims_intensity = intensity_model.cal_sims_intensity(self.predict_sims_emb, max_sims_len=self.sims_len,
#                                                                      num_units=self.num_units)
#
#         self.output()
#
#
# class MTAM_TPP_wendy(AttentionTPP_model):
#
#     def get_emb (self,target_time,time_last,time_now):
#         with tf.variable_scope('short-term', reuse=tf.AUTO_REUSE):
#
#
#             time_aware_gru_net_input = tf.concat([self.type_lst_embedding,  # TODO 这里加 time_lst_embedding
#                                               tf.expand_dims(time_last, 2),
#                                               tf.expand_dims(time_now, 2)],
#                                              axis=2)
#             self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
#                                                                           input_data=time_aware_gru_net_input,
#                                                                           input_length=tf.add(self.seq_len,-1),  #
#                                                                           type='new',
#                                                                           scope='gru')
#             short_term_intent = gather_indexes(batch_size=self.now_batch_size,
#                                            seq_length=self.max_seq_len,
#                                            width=self.num_units,
#                                            sequence_tensor=self.short_term_intent_temp,
#                                            positions=self.mask_index -1 )  #batch_size, num_units
#             short_term_intent4vallina = tf.expand_dims(short_term_intent, 1)
#         with tf.variable_scope('long-term',reuse=tf.AUTO_REUSE):
#             predict_lambda_emb = self.time_aware_attention.vanilla_attention(enc=self.type_lst_embedding,
#                                                                  dec=short_term_intent4vallina,
#                                                                  num_units=self.num_units,
#                                                                  num_heads=self.num_heads,
#                                                                  num_blocks=self.num_blocks,
#                                                                  dropout_rate=self.dropout_rate,
#                                                                  is_training=True,
#                                                                  reuse=False,
#                                                                  key_length=self.seq_len, # TODO 这里为什么不减去1
#                                                                  query_length=tf.ones_like(short_term_intent4vallina[:, 0, 0],
#                                                                                            dtype=tf.int32),
#                                                                  t_querys=tf.expand_dims(target_time,1),
#                                                                  t_keys=self.time_lst,
#                                                                  t_keys_length=self.max_seq_len,
#                                                                  t_querys_length=1
#                                                                  )
#             predict_lambda_emb = tf.reshape(predict_lambda_emb,[-1,self.num_units])
#         with tf.variable_scope('emb_for_time',reuse= tf.AUTO_REUSE):
#             #使用self-attention生成time的embedding L-1位的emb预测L位时间
#             time_self_att_output = self.attention.self_attention(enc=self.time_lst_embedding,
#                                                                  num_units=self.num_units,
#                                                                  num_heads=self.num_heads,
#                                                                  num_blocks=self.num_blocks,
#                                                                  dropout_rate=self.dropout_rate,
#                                                                  is_training=True,
#                                                                  reuse=None,
#                                                                  key_length = self.seq_len-1, # 只能看到seq_len-1这么长的记录
#                                                                  query_length=self.seq_len) # batch_size, seq_len, num_units
#             predict_emb_for_time = gather_indexes(batch_size=self.now_batch_size,
#                                            seq_length=self.max_seq_len,
#                                            width=self.num_units,
#                                            sequence_tensor=time_self_att_output,
#                                            positions=self.mask_index -1) # 取上一个时间的emb预测事件发生时间
#             # time_aware_gru_net_input2 = tf.concat([self.time_lst_embedding,  # TODO 这里加 time_lst_embedding
#             #                                       tf.expand_dims(time_last, 2),
#             #                                       tf.expand_dims(time_now, 2)],
#             #                                      axis=2)
#             # self.short_term_intent_temp2 = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
#             #                                                                   input_data=time_aware_gru_net_input2,
#             #                                                                   input_length=tf.add(self.seq_len, -1),  #
#             #                                                                   type='new',
#             #                                                                   scope='gru')
#             # predict_emb_for_time = gather_indexes(batch_size=self.now_batch_size,
#             #                                    seq_length=self.max_seq_len,
#             #                                    width=self.num_units,
#             #                                    sequence_tensor=self.short_term_intent_temp2,
#             #                                    positions=self.mask_index - 1)  # batch_size, num_units
#
#             # type_enc, type_att_vec = self.type_aware_attention.type_aware_vanilla_attention(enc = self.type_lst_embedding,
#             #                                                                       dec = tf.expand_dims(self.target_type_embedding,axis=1),
#             #                                                                       num_units=self.num_units,
#             #                                                                       num_heads=self.num_heads,
#             #                                                                       num_blocks=self.num_blocks,
#             #                                                                       dropout_rate=self.dropout_rate,
#             #                                                                       is_training=True,
#             #                                                                       reuse=None,
#             #                                                                       key_length=self.seq_len - 1, # 不考虑那个mask位了
#             #                                                                       query_length = tf.ones_like(self.seq_len),
#             #                                                                       )
#             #
#             # att_vec = tf.reshape(att_vec,shape= [-1,self.max_seq_len])
#
#         return layer_norm(predict_lambda_emb), predict_emb_for_time
#
#
#     def build_model(self):
#         self.time_aware_attention = Time_Aware_Attention()
#         self.attention = Attention()
#         self.gru_net_ins = GRU()
#         self.transformer_model = transformer_encoder()
#         self.type_aware_attention = Type_Aware_Attention()
#
#         predict_target_lambda_emb,predict_emb_for_time  = self.get_emb(self.target_time,self.target_time_last_lst,self.target_time_now_lst)
#
#         # sims_time_lst: batch_size, sims_len
#         predict_sims_emb= tf.zeros([self.now_batch_size,1])
#
#         sims_time = tf.squeeze(tf.split(self.sims_time_lst,self.sims_len,1),2)
#         sims_time_last = tf.squeeze(tf.split(self.sim_time_last_lst,self.sims_len,1),2)
#         sims_time_now = tf.squeeze(tf.split(self.sim_time_now_lst, self.sims_len, 1),2)
#         for i in range(self.sims_len):
#             #第i个时间 batch_size, num_units
#             cur_sims_emb, _ = self.get_emb(sims_time[i],sims_time_last[i],sims_time_now[i])
#             predict_sims_emb = tf.concat([predict_sims_emb,cur_sims_emb],axis = 1)
#
#         predict_sims_emb = predict_sims_emb[:,1:] # batch_size, sims_len * num_units
#         predict_sims_emb = tf.reshape(predict_sims_emb,[-1,self.sims_len,self.num_units])# batch_size, sims_len , num_units
#
#         self.predict_target_emb = predict_target_lambda_emb #
#         self.predict_sims_emb = predict_sims_emb
#         self.emb_for_time = predict_emb_for_time
#
#         with tf.variable_scope('prepare_emb'):
#             emb_for_intensity = self.predict_target_emb
#             emb_for_time = self.emb_for_time
#             emb_for_type = self.predict_target_emb
#
#
#         with tf.variable_scope('intensity_calculation'):
#             type_lookup_table = self.embedding.type_emb_lookup_table[:-3, :]  # type_num   , num_units
#             intensity_model = e_intensity(W=type_lookup_table, type_num=self.type_num)
#
#             self.target_lambda = intensity_model.cal_target_intensity(emb_for_intensity)
#             self.sims_lambda = intensity_model.cal_sims_intensity(self.predict_sims_emb, max_sims_len=self.sims_len,
#                                                                      num_units=self.num_units)
#
#         with tf.variable_scope('type_time_calculation'):
#             time_predictor = thp_time_predictor()
#             self.predict_time = time_predictor.predict_time(emb=emb_for_time,
#                                                             num_units=self.FLAGS.THP_M)  # batch_size, 1
#             # self.predict_time_interval = tf.reduce_sum(self.time_lst * emb_for_time, axis=1)# batch_size, 1
#             # col_idx = self.mask_index - 1
#             # row_idx = tf.reshape(tf.range(start=0, limit=self.now_batch_size, delta=1), [-1, 1])
#             # idx = tf.concat([row_idx, col_idx], axis=1)
#             # last_time = tf.gather_nd(self.time_lst, idx)
#             # self.predict_time = tf.reshape(self.predict_time_interval,[-1,1]) + tf.reshape(last_time,[-1,1])
#
#             type_predictor = thp_type_predictor()
#             self.predict_type_prob = type_predictor.predict_type(emb=emb_for_type,
#                                                                  num_units=self.FLAGS.THP_M,
#                                                                  type_num=self.type_num)
#         self.output()
#
#
# class MTAM_TPP_E(AttentionTPP_model):
#
#
#
#     def generate_emb(self,target_time,sims_time):
#         """
#         对于每一个训练样本，其实short-term都是一样的，
#         最后预测的emb不同主要是因为next_event_decoder这部分
#         :param: target_time # batch_size,
#         :param: sims_time # batch_size, sims_len
#         :return: batch_size, 1 + sims_len, num_units
#         """
#
#         target_time = tf.reshape(target_time,[-1,1])
#         target_sims_time = tf.concat([target_time, sims_time], axis = 1)
#
#         time_aware_attention = Time_Aware_Attention()
#         self.gru_net_ins = GRU()
#
#         with tf.variable_scope('type_history_encoder',reuse = tf.AUTO_REUSE):
#             history_emb = self.type_lst_embedding  # batch_size, max_seq_len, num_units
#
#         with tf.variable_scope('short_term_intent_encoder',reuse=tf.AUTO_REUSE):
#
#             tmp1 = tf.zeros_like(self.time_lst) # batch_size, seq_len
#             tmp2 = tf.zeros_like(self.time_lst) # batch_size, seq_len
#             time_aware_gru_net_input = tf.concat([history_emb, # TODO 为什么要加这个？
#                                                   tf.expand_dims(tmp1,2),
#                                                   tf.expand_dims(tmp2, 2)],
#                                                  axis = 2)
#
#
#             # TODO 为什么要把seq_len - 1 是为了得到索引吗
#             #sequence_len s是有效长度，不应该再减1
#             self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
#                                                                               input_data=time_aware_gru_net_input,
#                                                                               #input_length=tf.add(self.seq_len, -1),
#                                                                               input_length=tf.add(self.seq_len,-1), # TODO 是否需要-1
#                                                                               type='new')
#             # TODO mask_index实现 这个函数是干嘛的
#             #取到的应该是最后一个状态，因此mask_index 也不应该减1
#             # short_term_intent_temp (batch_size,seq_len,num_units)
#             short_term_intent = gather_indexes(batch_size=self.now_batch_size,
#                                                seq_length=self.max_seq_len,
#                                                width=self.num_units,
#                                                sequence_tensor=self.short_term_intent_temp,
#                                                positions=self.mask_index - 1) # TODO 感觉这里 应该是把最后一个取出来，而不是mask_index-1
#             # short_term_intent (batch_size, num_units)
#
#             short_term_intent4vallina = tf.expand_dims(short_term_intent, 1) # batch_size, 1, num_units
#             dup_short_term_intent4vallina = tf.tile(short_term_intent4vallina,[1,1+self.sims_len, 1]) # batch_size, 1+ sims_len, num_units
#
#         with tf.variable_scope('next_event_decoder',reuse=tf.AUTO_REUSE):
#
#             predict_emb = time_aware_attention.vanilla_attention(enc = history_emb,
#                                                                  dec = dup_short_term_intent4vallina,
#                                                                  num_units = self.num_units,
#                                                                  num_heads=self.num_heads,
#                                                                  num_blocks=self.num_blocks,
#                                                                  dropout_rate=self.dropout_rate,
#                                                                  is_training=True,
#                                                                  reuse=False,
#                                                                  key_length=self.seq_len,
#                                                                  query_length=tf.ones_like(
#                                                                      short_term_intent4vallina[:, 0, 0], # batch_size,
#                                                                      dtype=tf.int32) * (1 + self.sims_len),
#                                                                  t_querys=target_sims_time,
#                                                                  t_keys=self.time_lst,
#                                                                  t_keys_length=self.max_seq_len,
#                                                                  t_querys_length= 1 + self.sims_len
#                                                                  )
#             predict_emb = layer_norm(predict_emb) # batch_size, 1+sims_len , num_units
#
#         return predict_emb
#
#
#     def build_model(self):
#
#
#
#         with tf.variable_scope('intensity_calculation',reuse=tf.AUTO_REUSE):
#             intensity_fun = e_intensity(W = self.embedding.type_emb_lookup_table,
#                                         type_num = self.type_num)
#
#
#
#             predict_emb = self.generate_emb(target_time=self.target_time,
#                                             sims_time=self.sims_time_lst)  # batch_size, 1+sims_len, num_units
#
#             predict_emb = tf.reshape(predict_emb,
#                                      [-1, predict_emb.shape[-1]])  # batch_size * (1+sims_len), num_units
#
#             predict_intensity = intensity_fun.cal_intensity(predict_emb)
#             predict_intensity = tf.reshape(predict_intensity,[-1,1+self.sims_len,self.type_num]) # batch_size, 1+sims_len, num_units
#
#             predict_target_intensity = predict_intensity[:,0,:] # batch_size, num_units
#             predict_sims_intensity = predict_intensity[:,1:,:]
#
#
#             self.lambda_prob = predict_target_intensity  # batch_size, type_num]
#             # TODO 计算积分项
#
#
#
#         self.output()
#
#
#
#
#
#
#
#
# class MTAM_TPP_W(AttentionTPP_model):
#
#
#
#     def generate_emb(self,target_time,sims_time):
#         """
#         对于每一个训练样本，其实short-term都是一样的，
#         最后预测的emb不同主要是因为next_event_decoder这部分
#         :param: target_time # batch_size,
#         :param: sims_time # batch_size, sims_len
#         :return: batch_size, 1 + sims_len, num_units
#         """
#
#         target_time = tf.reshape(target_time,[-1,1])
#         target_sims_time = tf.concat([target_time, sims_time], axis = 1)
#
#         time_aware_attention = Time_Aware_Attention()
#         self.gru_net_ins = GRU()
#
#         with tf.variable_scope('type_history_encoder',reuse = tf.AUTO_REUSE):
#             history_emb = self.type_lst_embedding  # batch_size, max_seq_len, num_units
#
#         with tf.variable_scope('short_term_intent_encoder',reuse=tf.AUTO_REUSE):
#
#             tmp1 = tf.zeros_like(self.time_lst) # batch_size, seq_len
#             tmp2 = tf.zeros_like(self.time_lst) # batch_size, seq_len
#             time_aware_gru_net_input = tf.concat([history_emb, # TODO 为什么要加这个？
#                                                   tf.expand_dims(tmp1,2),
#                                                   tf.expand_dims(tmp2, 2)],
#                                                  axis = 2)
#
#
#             # TODO 为什么要把seq_len - 1 是为了得到索引吗
#             self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
#                                                                               input_data=time_aware_gru_net_input,
#                                                                               # input_length=tf.add(self.seq_len, -1),
#                                                                               input_length=tf.add(self.seq_len,-1), # TODO 是否需要-1
#                                                                               type='new')
#             # TODO mask_index实现 这个函数是干嘛的
#             # short_term_intent_temp (batch_size,seq_len,num_units)
#             short_term_intent = gather_indexes(batch_size=self.now_batch_size,
#                                                seq_length=self.max_seq_len,
#                                                width=self.num_units,
#                                                sequence_tensor=self.short_term_intent_temp,
#                                                positions=self.mask_index -1 ) # TODO 感觉这里 应该是把最后一个取出来，而不是mask_index-1
#             # short_term_intent (batch_size, num_units)
#
#             short_term_intent4vallina = tf.expand_dims(short_term_intent, 1) # batch_size, 1, num_units
#             dup_short_term_intent4vallina = tf.tile(short_term_intent4vallina,[1,1+self.sims_len, 1]) # batch_size, 1+ sims_len, num_units
#
#         with tf.variable_scope('next_event_decoder',reuse=tf.AUTO_REUSE):
#
#             predict_emb = time_aware_attention.vanilla_attention(enc = history_emb,
#                                                                  dec = dup_short_term_intent4vallina,
#                                                                  num_units = self.num_units,
#                                                                  num_heads=self.num_heads,
#                                                                  num_blocks=self.num_blocks,
#                                                                  dropout_rate=self.dropout_rate,
#                                                                  is_training=True,
#                                                                  reuse=False,
#                                                                  key_length=self.seq_len,
#                                                                  query_length=tf.ones_like(
#                                                                      short_term_intent4vallina[:, 0, 0], # batch_size,
#                                                                      dtype=tf.int32) * (1 + self.sims_len),
#                                                                  t_querys=target_sims_time,
#                                                                  t_keys=self.time_lst,
#                                                                  t_keys_length=self.max_seq_len,
#                                                                  t_querys_length= 1 + self.sims_len
#                                                                  )
#             predict_emb = layer_norm(predict_emb) # batch_size, 1+sims_len , num_units
#
#         return predict_emb
#
#
#     def build_model(self):
#
#
#         with tf.variable_scope('intensity_calculation',reuse=tf.AUTO_REUSE):
#             intensity_fun = mlt_intensity()
#
#             layer_units = self.FLAGS.layers
#
#             target_demo = tf.reshape(self.seq_len, [-1, 1])  # batch_size, 1
#             sims_demo = tf.reshape(tf.tile(self.seq_len, [self.sims_len]),
#                                    [-1, self.sims_len, 1])  # batch_size , sims_len,1
#
#             predict_target_intensity = tf.zeros_like(target_demo, dtype=tf.float32)  # batch_size, 1
#             predict_sims_intensity = tf.zeros_like(sims_demo, dtype=tf.float32)  # batch_size, sims_len , 1
#             predict_sims_intensity = tf.reshape(predict_sims_intensity, [-1, 1])
#
#             predict_emb = self.generate_emb(target_time=self.target_time,
#                                             sims_time=self.sims_time_lst)  # batch_size, 1+sims_len, num_units
#
#             predict_emb = tf.reshape(predict_emb,
#                                      [-1, predict_emb.shape[-1]])  # batch_size * (1+sims_len), num_units
#
#             for type in range(self.type_num):
#
#
#                 predict_type_intensity = intensity_fun.cal_type_intensity(emb=predict_emb,
#                                                                           type=type,
#                                                                           layer_units=layer_units)
#                 predict_type_intensity = tf.reshape(predict_type_intensity, [-1, 1 + self.sims_len, 1])
#
#                 predict_target_type_intensity = predict_type_intensity[:, 0, :]  # batch_size, 1
#                 predict_sims_type_intensity = predict_type_intensity[:, 1:, :]
#                 predict_sims_type_intensity = tf.reshape(predict_sims_type_intensity,
#                                                          [-1, 1])  # batch_size * sims_len , 1
#
#                 predict_target_intensity = tf.concat([predict_target_intensity, predict_target_type_intensity], axis=1)
#                 predict_sims_intensity = tf.concat([predict_sims_intensity, predict_sims_type_intensity], axis=1)
#
#             self.lambda_prob = predict_target_intensity[:, 1:]  # batch_size, type_num]
#             # TODO 计算积分项
#
#
#
#         self.output()
#
#
#
#
#
#
#
#
#
#
#
# """
# -------以下是检索模型----------
#
# """
#
# class AttentionTPP_MLT(AttentionTPP_model):
#     """
#     每种类型用不同的全连接层预测
#     """
#
#
#     def generate_emb(self,type_lst_emb, time_lst, target_type, target_time, seq_len, sims_lst):
#         """
#
#         :param type_lst_emb: batch_size, max_seq_len, type_emb_units
#         :param time_lst:  batch_size, max_seq_len
#         :param target_type: batch_size,
#         :param target_time: batch_size,
#         :param seq_len: batch_size,
#         :param sims_lst: batch_size, sims_len
#         :return: batch_size, 1 + sims_len, num_units 经过attention后得到的embedding
#         """
#
#         target_time = tf.reshape(target_time,[-1,1]) # batch_size, 1
#         target_sims_time = tf.concat([target_time,sims_lst], axis = 1) # batch_size, 1+ sims_len
#
#         target_type = tf.reshape(target_type, [-1,1])
#         target_type_emb = self.embedding.get_type_embedding(target_type) # batch_size, 1, type_emb_units
#         target_sims_type_emb = tf.tile(target_type_emb, [1,self.sims_len + 1, 1])
#
#         query_len = tf.ones_like(seq_len, dtype = tf.int32) * (self.sims_len + 1)
#
#         with tf.variable_scope('target_decoder', reuse = tf.AUTO_REUSE):
#             time_aware_attention = Time_Aware_Attention()
#
#             next_emb = time_aware_attention.vanilla_attention(enc = type_lst_emb,
#                                                                dec = target_sims_type_emb,
#                                                                num_units = self.num_units,
#                                                                num_heads=self.num_heads,
#                                                                num_blocks=self.num_blocks,
#                                                                dropout_rate=self.dropout_rate,
#                                                                is_training=True,
#                                                                reuse = True,# TODO 这里写true还是false
#                                                                key_length=seq_len,
#                                                                query_length = query_len,
#                                                                t_querys=target_sims_time,
#                                                                t_keys = time_lst,
#                                                                t_keys_length=self.max_seq_len,
#                                                                    t_querys_length = self.sims_len + 1)
#             predict_emb = layer_norm(next_emb)
#         # predict_emb = tf.reshape(next_emb,[-1,self.sims_len + 1, self.num_units]) # batch_size, 1+sims_len, num_units
#         return predict_emb
#
#
#
#     def build_model(self):
#
#         with tf.variable_scope('intensity_calculation', reuse = tf.AUTO_REUSE):
#
#             intensity_fun = mlt_intensity()
#
#             layer_units = self.FLAGS.layers
#
#             target_demo = tf.reshape(self.seq_len,[-1,1]) # batch_size, 1
#             sims_demo = tf.reshape(tf.tile(self.seq_len, [self.sims_len]) ,[-1,self.sims_len,1])# batch_size , sims_len,1
#             predict_target_intensity = tf.zeros_like(target_demo, dtype = tf.float32) # batch_size, 1
#             predict_sims_intensity = tf.zeros_like(sims_demo, dtype = tf.float32) # batch_size, sims_len , 1
#             predict_sims_intensity = tf.reshape(predict_sims_intensity,[-1,1])
#
#             for type in range(self.type_num):
#                 target_type = tf.ones_like(self.seq_len) * type # batch_size,
#
#                 predict_emb = self.generate_emb(type_lst_emb= self.type_lst_embedding,
#                                                 time_lst = self.time_lst,
#                                                 target_type=target_type,
#                                                 target_time=self.target_time,
#                                                 sims_lst=self.sims_time_lst,
#                                                 seq_len = self.seq_len) # batch_size, 1+sims_len, num_units
#
#                 predict_emb = tf.reshape(predict_emb, [-1, predict_emb.shape[-1]]) # batch_size * (1+sims_len), num_units
#
#                 predict_type_intensity = intensity_fun.cal_type_intensity(emb = predict_emb,
#                                                                           type = type,
#                                                                           layer_units= layer_units)
#                 predict_type_intensity = tf.reshape(predict_type_intensity,[-1,1+self.sims_len, 1])
#
#                 predict_target_type_intensity = predict_type_intensity[:,0,:] # batch_size, 1
#                 predict_sims_type_intensity =predict_type_intensity[:,1:,:]
#                 predict_sims_type_intensity = tf.reshape(predict_sims_type_intensity,[-1,1]) # batch_size * sims_len , 1
#
#                 predict_target_intensity = tf.concat([predict_target_intensity, predict_target_type_intensity], axis=1)
#                 predict_sims_intensity = tf.concat([predict_sims_intensity, predict_sims_type_intensity], axis = 1)
#
#
#             self.lambda_prob = predict_target_intensity[:,1:] # batch_size, type_num
#         self.output()
#
#
#
#
#
# class AttentionTPP(AttentionTPP_model):
#     """
#     上层使用一个全连接层同时预测多个类型概率
#     """
#
#
#     def generate_emb(self,type_lst_emb, time_lst, target_type, target_time, seq_len, sims_lst):
#         """
#
#         :param type_lst_emb: batch_size, max_seq_len, type_emb_units
#         :param time_lst:  batch_size, max_seq_len
#         :param target_type: batch_size,
#         :param target_time: batch_size,
#         :param seq_len: batch_size,
#         :param sims_lst: batch_size, sims_len
#         :return: batch_size, 1 + sims_len, num_units 经过attention后得到的embedding
#         """
#
#         target_time = tf.reshape(target_time,[-1,1]) # batch_size, 1
#         target_sims_time = tf.concat([target_time,sims_lst], axis = 1) # batch_size, 1+ sims_len
#
#         target_type = tf.reshape(target_type, [-1,1])
#         target_type_emb = self.embedding.get_type_embedding(target_type) # batch_size, 1, type_emb_units
#         target_sims_type_emb = tf.tile(target_type_emb, [1,self.sims_len + 1, 1])
#
#         query_len = tf.ones_like(seq_len, dtype = tf.int32) * (self.sims_len + 1)
#
#         with tf.variable_scope('target_decoder', reuse = tf.AUTO_REUSE):
#             time_aware_attention = Time_Aware_Attention()
#
#             next_emb = time_aware_attention.vanilla_attention(enc = type_lst_emb,
#                                                                dec = target_sims_type_emb,
#                                                                num_units = self.num_units,
#                                                                num_heads=self.num_heads,
#                                                                num_blocks=self.num_blocks,
#                                                                dropout_rate=self.dropout_rate,
#                                                                is_training=True,
#                                                                reuse = True,# TODO 这里写true还是false
#                                                                key_length=seq_len,
#                                                                query_length = query_len,
#                                                                t_querys=target_sims_time,
#                                                                t_keys = time_lst,
#                                                                t_keys_length=self.max_seq_len,
#                                                                t_querys_length = self.sims_len + 1)
#             predict_emb = layer_norm(next_emb)
#         # predict_emb = tf.reshape(next_emb,[-1,self.sims_len + 1, self.num_units]) # batch_size, 1+sims_len, num_units
#         return predict_emb
#
#
#
#     def build_model(self):
#
#         with tf.variable_scope('intensity_calculation', reuse = tf.AUTO_REUSE):
#
#             intensity_fun = single_intensity()
#
#             layer_units = self.FLAGS.layers
#
#
#
#             predict_emb = self.generate_emb(type_lst_emb=self.type_lst_embedding,
#                                                          time_lst = self.time_lst,
#                                                          target_type = self.target_type,
#                                                          target_time= self.target_time,
#                                                          sims_lst = self.sims_time_lst,
#                                                          seq_len = self.seq_len)
#
#             predict_target_intensity = intensity_fun.cal_intensity(emb=predict_emb,
#                                                                    layer_units=layer_units,
#                                                                    scope='intensity_calculation',
#                                                                    num=self.type_num)
#
#             self.lambda_prob = predict_target_intensity[:,0,:] # batch_size, type_num
#         self.output()
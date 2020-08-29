# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 18:38
# @Author  : zxl
# @FileName: THP.py

import tensorflow as tf
from Model.Modules.intensity_calculation import thp_intensity_calculation
from Model.base_model import base_model
from Model.Modules.transformer_encoder import transformer_encoder
from Model.Modules.net_utils import gather_indexes, layer_norm


class THP_model(base_model):
    def __init__(self, FLAGS, Embedding, sess):

        super(THP_model,self).__init__(FLAGS= FLAGS,
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

        self.type_lst_embedding, \
        self.time_lst, \
        self.time_lst_embedding, \
        self.target_type_embedding, \
        self.target_type,\
        self.target_time, \
        self.seq_len, \
        self.T_lst,\
        self.sims_time_lst,\
        self.target_time_last_lst,\
        self.target_time_now_lst, \
        self.sim_time_last_lst, \
        self.sim_time_now_lst = self.embedding.get_embedding(self.type_emb_size)
        self.mask_index = tf.reshape(self.seq_len - 1, [-1, 1])

        self.build_model()
        self.init_variables(sess)


class THP(THP_model):


    def hidden_emb_generation(self,S,M,MH,scope = 'hidden_emb_encoding'):
        """

        :param S: # batch_size, seq_len M
        :param M:
        :param MH:
        :param scope:
        :return: batch_size, seq_len, M
        """
        with tf.variable_scope(scope):
            W1 = tf.get_variable('S_W1', shape=(M, MH))
            W2 = tf.get_variable('S_W2', shape=(MH, M))
            b1 = tf.get_variable('S_b1', shape=(MH,))
            b2 = tf.get_variable('S_b2', shape=(M,))

            flatten_S = tf.reshape(S, [-1,M])
            H1 = tf.nn.relu(tf.matmul(flatten_S,W1) + b1)
            H = tf.matmul(H1,W2) + b2
        return H




    def build_model(self):

        transformer_model = transformer_encoder()

        with tf.variable_scope('history_encoding',reuse=tf.AUTO_REUSE):
            X = self.type_lst_embedding + self.time_lst_embedding

        with tf.variable_scope('transformer_encoding',reuse=tf.AUTO_REUSE):
            S = transformer_model.stack_multihead_self_attention(stack_num=self.FLAGS.THP_stack_num,
                                                                 X = X,
                                                                 M = self.FLAGS.THP_M,
                                                                 Mk = self.FLAGS.THP_Mk,
                                                                 Mv= self.FLAGS.THP_Mv,
                                                                 L = self.max_seq_len,
                                                                 N = self.now_batch_size,
                                                                 head_num=self.FLAGS.THP_head_num,
                                                                 ) # batch_size, seq_len, M
        with tf.variable_scope('hidden_emb_cal',reuse=tf.AUTO_REUSE):
            M = self.FLAGS.THP_M
            MH = self.FLAGS.THP_MH
            H = self.hidden_emb_generation(S=S, M = M, MH = MH) # batch_size, seq_len, M

            discrete_emb = gather_indexes(batch_size=self.now_batch_size,
                           seq_length=self.max_seq_len,
                           width=M,
                           sequence_tensor=H,
                           positions=self.mask_index) # batch_size, M

        with tf.variable_scope('lambda_calculation',reuse=tf.AUTO_REUSE):
            col_idx = self.mask_index - 1
            row_idx = tf.reshape(tf.range(start=0, limit=self.now_batch_size, delta=1), [-1, 1])
            idx = tf.concat([row_idx, col_idx], axis=1)

            last_time = tf.gather_nd(self.time_lst, idx) + 1e-9
            self.last_time =  last_time    # TODO for testing

            intensity_model = thp_intensity_calculation()

            self.target_lambda = intensity_model.cal_target_intensity(hidden_emb=discrete_emb,
                                                                   target_time=self.target_time,
                                                                   last_time=last_time,
                                                                   type_num=self.type_num)
            self.sims_lambda = intensity_model.cal_sims_intensity(hidden_emb=discrete_emb,
                                                                  sims_time = self.sims_time_lst,
                                                                  last_time= last_time,
                                                                  sims_len = self.sims_len,
                                                                  type_num = self.type_num)
            self.predict_target_emb = discrete_emb
        self.output()



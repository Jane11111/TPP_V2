# -*- coding: utf-8 -*-
# @Time    : 2020/9/11 10:08
# @Author  : zxl
# @FileName: RMTPP.py


import tensorflow as tf
from Model.base_model import base_model
from Model.Modules.transformer_encoder import transformer_encoder
from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.Modules.time_prediction import thp_time_predictor
from Model.Modules.type_prediction import thp_type_predictor
from Model.Modules.rmtpp_rnn import Rmtpp_RNN
from tensorflow.python.ops import array_ops
from Model.Modules.intensity_calculation import rmtpp_density_calculation

class RMTPP_model(base_model):
    def __init__(self, FLAGS, Embedding, sess):

        super(RMTPP_model,self).__init__(FLAGS= FLAGS,
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
        self.not_first_lst,\
        self.sims_time_lst,\
        self.target_time_last_lst,\
        self.target_time_now_lst, \
        self.sim_time_last_lst, \
        self.sim_time_now_lst = self.embedding.get_embedding(self.type_emb_size)
        self.mask_index = tf.reshape(self.seq_len - 1, [-1, 1])

        self.build_model()
        self.init_variables(sess)

class RMTPP(RMTPP_model):

    def get_state(self, time_last):
        with tf.variable_scope('cstm_get_emb', reuse=tf.AUTO_REUSE):
            ctsm_input = tf.concat([self.type_lst_embedding,
                                    tf.expand_dims(time_last, 2)],
                                   axis=2)
            output,state = self.rmtpp_rnn_model.rmtpp_net(hidden_units=self.num_units,
                                              input_data=ctsm_input,
                                              input_length=tf.add(self.seq_len, -1))
            h_i_minus = state[0]
        return  h_i_minus

    def get_last_timestamp(self):
        col_idx = self.mask_index - 1
        row_idx = tf.reshape(tf.range(start=0, limit=self.now_batch_size, delta=1), [-1, 1])
        idx = tf.concat([row_idx, col_idx], axis=1)
        last_time = tf.gather_nd(self.time_lst, idx)
        return last_time

    def build_model(self):

        self.rmtpp_rnn_model = Rmtpp_RNN()

        h_i_minus = self.get_state(self.target_time_last_lst)
        last_time = self.get_last_timestamp()
        self.test = h_i_minus


        with tf.variable_scope('density_calculation', reuse=tf.AUTO_REUSE):

            rmtpp_density_model = rmtpp_density_calculation()
            self.f_t = rmtpp_density_model.cal_target_density(hidden_emb=h_i_minus,
                                                         target_time = self.target_time,
                                                         last_time=last_time)

        with tf.variable_scope('time_type_prediction'):
            # TODO 是否需要用积分计算时间？
            time_predictor = thp_time_predictor()
            self.predict_time = time_predictor.predict_time(emb=h_i_minus,
                                                            num_units=self.num_units) # batch_size, 1

            type_predictor = thp_type_predictor()
            self.predict_type_prob = type_predictor.predict_type(emb=h_i_minus,
                                                                 num_units=self.num_units,
                                                                 type_num=self.type_num) # batch_size, type_num
        self.output()

    def output(self):

        with tf.name_scope('loss_function'):

            self.l2_norm = tf.add_n([
                tf.nn.l2_loss(self.type_lst_embedding)
            ])
            regulation_rate = self.FLAGS.regulation_rate
            one_hot_type = tf.one_hot(
                self.target_type, depth = self.FLAGS.type_num, dtype = tf.float32
            )
            one_hot_type = tf.reshape(one_hot_type,[-1,self.FLAGS.type_num ]) # batch_size, type_num



            """type"""
            self.predict_type_prob = tf.nn.softmax(self.predict_type_prob)
            log_probs = tf.log (self.predict_type_prob + 1e-9)
            self.cross_entropy_loss = -tf.reduce_sum(log_probs * one_hot_type, axis=[-1])  # batch_size,

            """time"""
            self.SE_loss  = (self.target_time-tf.squeeze(self.predict_time)) ** 2  # batch_size,

            """lambda"""


            self.log_likelihood =  tf.squeeze(tf.log(self.f_t+1e-9)) # batch_size,


            self.log_likelihood_loss = -self.log_likelihood

            self.time_likelohood = tf.constant([0,0])
            self.type_likelihood = tf.constant([0,0])


            self.loss =  self.llh_decay_rate * tf.reduce_mean(self.log_likelihood_loss) \
                        + tf.reduce_mean(self.cross_entropy_loss) # - (logP(y|h) + logf(d|h))

            # for metrics
            self.labels = one_hot_type
            # self.target_lambda = target_lambda

            tf.summary.scalar('l2_norm', self.l2_norm)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('llh_decay_rate', self.llh_decay_rate)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('seq_log_likelihood', tf.reduce_mean(self.log_likelihood) )
            tf.summary.scalar('cross_entropy_loss', tf.reduce_mean(self.cross_entropy_loss) )
            tf.summary.scalar('sqrt_mean_square_error_loss', tf.sqrt(tf.reduce_mean(self.SE_loss) ))
        self.cal_gradient(tf.trainable_variables())












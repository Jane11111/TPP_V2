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
            intensity_model = hp_intensity_calculation( )

            self.target_lambda = intensity_model.cal_target_intensity(timenow_lst=self.target_time_now_lst,
                                                                      type_lst=self.type_lst,
                                                                      type_num = self.type_num,
                                                                      seq_len = self.seq_len,
                                                                      max_seq_len = self.max_seq_len
                                                                      ) # batch_size, type_num
            # self.sims_lambda = intensity_model.cal_sims_intensity(sims_timenow_lst = self.sim_time_now_lst,
            #                                                       type_lst=self.type_lst,
            #                                                       type_num=self.type_num,
            #                                                       seq_len=self.seq_len,
            #                                                       max_seq_len=self.max_seq_len,
            #                                                       sims_len=self.sims_len)
            last_time = tf.squeeze(gather_indexes(batch_size=self.now_batch_size,
                                       seq_length=self.max_seq_len,
                                       width=1,
                                       sequence_tensor=self.time_lst,
                                       positions=self.mask_index - 1) ) # 上一个时间
            self.integral_lambda  = intensity_model.cal_integral_intensity(t_last = last_time,
                                                                          t_target=self.target_time,
                                                                          time_lst = self.time_lst,
                                                                          type_lst = self.type_lst,
                                                                          type_num=self.type_num,
                                                                          seq_len=self.seq_len,
                                                                          max_seq_len=self.max_seq_len)
            # self.test = self.target_lambda
            # self.f_t = self.target_lambda * tf.exp(-self.integral_lambda)

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
            log_probs = tf.log (self.predict_type_prob + 0.1)
            self.cross_entropy_loss = -tf.reduce_sum(log_probs * one_hot_type, axis=[-1])  # batch_size,

            """time"""
            self.SE_loss  = (self.target_time-tf.squeeze(self.predict_time)) ** 2  # batch_size,

            """lambda"""
            lambda_k_t= self.target_lambda * one_hot_type
            self.lambda_k_t= tf.reduce_sum(lambda_k_t,axis=1,keep_dims=True) # batch_size, 1

            self.test = [self.lambda_k_t, self.integral_lambda]

            self.log_likelihood = tf.log(self.lambda_k_t+1e-9)-self.integral_lambda

            self.log_likelihood =  tf.squeeze(self.log_likelihood) # batch_size,


            self.log_likelihood_loss = -self.log_likelihood

            self.time_likelohood = tf.constant([0,0])
            self.type_likelihood = tf.constant([0,0])


            self.loss =  tf.reduce_mean(self.SE_loss) \
                         + self.llh_decay_rate * tf.reduce_mean(self.log_likelihood_loss) \
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



class IHP(HP_model):


    def build_model(self):



        with tf.variable_scope('intensity_calculation', reuse=tf.AUTO_REUSE):
            intensity_model = ihp_intensity_calculation( )

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



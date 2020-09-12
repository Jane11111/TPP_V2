# -*- coding: utf-8 -*-
# @Time    : 2020/9/11 16:45
# @Author  : zxl
# @FileName: SAHP.py

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from Model.Modules.intensity_calculation import sahp_intensity_calculation
from Model.base_model import base_model
from Model.Modules.transformer_encoder import transformer_encoder
from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.Modules.time_prediction import thp_time_predictor
from Model.Modules.type_prediction import thp_type_predictor
from Model.Modules.sahp_self_attention import SahpSelfAttention

class SAHP_model(base_model):
    def __init__(self, FLAGS, Embedding, sess):

        super(SAHP_model,self).__init__(FLAGS= FLAGS,
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


class SAHP(SAHP_model):

    def get_emb(self,time_emb,type_emb):
        """

        :param time_emb: batch_size, seq_len, num_units
        :param type_emb: batch_size, seq_len, num_units
        :return:
        """

        X = time_emb + type_emb
        M = self.FLAGS.THP_M
        Mv = self.FLAGS.THP_Mv
        L = self.max_seq_len
        hidden_emb = self.sahp_att_model.multistack_multihead_self_attention(X = X,
                                                                             M = M,
                                                                             Mv= Mv,
                                                                             L = L,
                                                                             N = self.now_batch_size,
                                                                             head_num=self.FLAGS.THP_head_num,
                                                                             stack_num=self.FLAGS.THP_stack_num,
                                                                             dropout_rate=self.dropout_rate,
                                                                             )
        h = gather_indexes(batch_size=self.now_batch_size,
                           seq_length=self.max_seq_len,
                           width=M,
                           sequence_tensor=hidden_emb,
                           positions=self.mask_index-1) # 取上一个时刻的emb
        return h

    # def gelu(self,input_tensor):
    #     cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    #     return input_tensor * cdf

    def gelu(self,features, approximate=False, name=None):
        """Compute the Gaussian Error Linear Unit (GELU) activation function.
        Gaussian error linear unit (GELU) computes
        `x * P(X <= x)`, where `P(X) ~ N(0, 1)`.
        The (GELU) nonlinearity weights inputs by their value, rather than gates
        inputs by their sign as in ReLU.
        For example:

        array([-0.00404951, -0.15865529,  0.        ,  0.8413447 ,  2.9959507 ],
            dtype=float32)

        array([-0.00363752, -0.15880796,  0.        ,  0.841192  ,  2.9963627 ],
            dtype=float32)
        Args:
          features: A `Tensor` representing preactivation values.
          approximate: An optional `bool`. Defaults to `False`. Whether to enable
            approximation.
          name: A name for the operation (optional).
        Returns:
          A `Tensor` with the same type as `features`.
        References:
          [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).
        """
        with ops.name_scope(name, "Gelu", [features]):
            features = ops.convert_to_tensor(features, name="features")
            if approximate:
                coeff = math_ops.cast(0.044715, features.dtype)
                return 0.5 * features * (
                        1.0 + math_ops.tanh(0.7978845608028654 *
                                            (features + coeff * math_ops.pow(features, 3))))
            else:
                return 0.5 * features * (1.0 + math_ops.erf(
                    features / math_ops.cast(1.4142135623730951, features.dtype)))
    def build_model(self):

        self.sahp_att_model = SahpSelfAttention()

        h = self.get_emb(self.time_lst_embedding,self.type_lst_embedding) # TODO time_lst_embedidng 需要更改

        M = self.FLAGS.THP_M
        dtype = h.dtype
        with tf.variable_scope('prepare_parameter',reuse=tf.AUTO_REUSE):

            W_mu = tf.get_variable('W_mu',shape = (M,1),dtype = dtype)
            W_eta = tf.get_variable('W_eta',shape=(M,1), dtype=dtype)
            W_gamma = tf.get_variable('W_gamma', shape=(M,1), dtype=dtype)

            mu = self.gelu(tf.matmul(h,W_mu)) # TODO 这个函数需要检查
            eta = self.gelu(tf.matmul(h,W_eta))
            gamma = tf.nn.softplus(tf.matmul(h,W_gamma))


        with tf.variable_scope('intensity_calculation', reuse=tf.AUTO_REUSE):
            last_time = tf.squeeze(gather_indexes(batch_size=self.now_batch_size,
                                 seq_length=self.max_seq_len,
                                 width=1,
                                 sequence_tensor=tf.expand_dims(self.time_lst, axis=-1),
                                 positions=self.mask_index-1)) # target_time 上个时刻

            intensity_model = sahp_intensity_calculation(mu = mu,
                                                         eta = eta,
                                                         gamma = gamma)

            self.target_lambda = intensity_model.cal_target_intensity(target_time=self.target_time,
                                                                      last_time=last_time,
                                                                      type_num=self.type_num)
            self.sims_lambda = intensity_model.cal_sims_intensity(sims_time=self.sims_time_lst,
                                                                  last_time=last_time,
                                                                  sims_len=self.sims_len,
                                                                  type_num=self.type_num)

        with tf.variable_scope('predict_time_type',reuse=tf.AUTO_REUSE):
            time_predictor = thp_time_predictor()
            self.predict_time = time_predictor.predict_time(emb=h,
                                                            num_units=self.FLAGS.THP_M)  # batch_size, 1
            type_predictor = thp_type_predictor()
            self.predict_type_prob = type_predictor.predict_type(emb=h,
                                                                 num_units=self.FLAGS.THP_M,
                                                                 type_num=self.type_num)  # batch_size, type_num

        self.output()










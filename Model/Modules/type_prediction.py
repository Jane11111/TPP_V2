# -*- coding: utf-8 -*-
# @Time    : 2020/8/29 16:21
# @Author  : zxl
# @FileName: type_prediction.py


import tensorflow as tf


class thp_type_predictor():

    def __init__(self):
        pass

    def predict_type(self, emb, num_units, type_num,scope='type_prediction'):
        """

        :param emb: batch_size, num_units
        :return:
        """

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            W_type = tf.get_variable('W_time', shape=(num_units, type_num))

            types = tf.matmul(emb, W_type)
            predict_type_probs = tf.nn.softmax(types)
        return predict_type_probs
# -*- coding: utf-8 -*-
# @Time    : 2020/8/29 16:16
# @Author  : zxl
# @FileName: time_prediction.py

import tensorflow as tf

class thp_time_predictor():

    def __init__(self):
        pass

    def predict_time(self,emb,num_units,scope = 'time_prediction'):
        """

        :param emb: batch_size, num_units
        :return:
        """

        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):

            W_time = tf.get_variable('W_time',shape = (num_units,1))

            times = tf.matmul(emb,W_time)
        return times
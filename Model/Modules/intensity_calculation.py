# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 11:02
# @Author  : zxl
# @FileName: intensity_calculation.py

import tensorflow as tf

class intensity_base(object):

    def __init__(self):
        pass


    def cal_intensity(self, emb, layer_units, scope):
        """

        :param emb: 当前type time作为query，历史type time 作为key得到的emb
        :param layer_units: 计算强度的神经网络每一层神经单元个数
        :param scope:
        :return:
        """

        with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
            input_data = emb
            for unit_num in layer_units:
                H = tf.layers.dense(input_data,
                                    units = unit_num,
                                    activation = tf.nn.relu)
                input_data = H
            outputs = tf.layers.dense(input_data,
                                      units = 1,
                                      activation = tf.nn.softplus)
        return outputs


class mlt_intensity(intensity_base):
    """
    不同type的强度，实用不同的网络结构
    """

    def __init__(self):
        super(mlt_intensity, self).__init__()


    def cal_type_intensity(self, emb, type, layer_units):
        """

        :param emb:
        :param type: 当前的type编号
        :param layer_units: 计算intensity的网络结构
        :return:
        """

        scope = 'type_intensity_calculation_' + str(type)
        predict_type_intensity = self.cal_intensity(emb, layer_units, scope)
        return predict_type_intensity




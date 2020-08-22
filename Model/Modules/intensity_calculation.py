# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 11:02
# @Author  : zxl
# @FileName: intensity_calculation.py

import tensorflow as tf

class intensity_base(object):

    def __init__(self):
        pass


    def cal_intensity(self, emb, layer_units, scope, num):
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
                                      units = num,
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
        predict_type_intensity = self.cal_intensity(emb, layer_units, scope,1)
        return predict_type_intensity



class single_intensity(intensity_base):
    """
    不同type的强度，实用不同的网络结构
    """

    def __init__(self):
        super(single_intensity, self).__init__()


    def cal_type_intensity(self, emb,  layer_units, type_num):
        """

        :param emb:
        :param type: 当前的type编号
        :param layer_units: 计算intensity的网络结构
        :return:
        """

        scope = 'type_intensity_calculation_' + str(type)
        predict_type_intensity = self.cal_intensity(emb, layer_units, scope,type_num)
        return predict_type_intensity


class e_intensity():
    """
    使用type table 计算强度
    """

    def __init__(self,W,type_num):
        """

        :param W: type_num, num_units
        """
        self.type_num = type_num
        self.W = W


    def cal_intensity(self,emb):

        """

        :param emb: batch_size, num_units
        :return:  batch_size, type_num
        """

        intensity = tf.matmul(emb, self.W, transpose_b= True)
        return intensity

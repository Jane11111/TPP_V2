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

        intensity = tf.nn.relu(tf.matmul(emb, self.W, transpose_b= True))
        return intensity

    def cal_target_intensity(self,emb):
        """

        :param emb:
        :return:
        """
        return self.cal_intensity(emb)

    def cal_sims_intensity(self,emb,max_sims_len,num_units):
        """

        :param emb: batch_size, max_sims_len, num_units
        :return:
        """
        emb = tf.reshape(emb,[-1,num_units])
        intensity = self.cal_intensity(emb)
        intensity = tf.reshape(intensity,[-1,max_sims_len, self.type_num])
        return intensity



class thp_intensity_calculation():

    def __init__(self):
        pass

    def cal_intensity(self,hidden_emb,time_interval,last_time,type):
        """

        :param hidden_emb: N, num_units
        :param time_interval: N,1
        :param last_time: N,1
        :param type: int
        :return:
        """
        with tf.variable_scope('intensity_calculation'):
            num_units = hidden_emb.shape[1]
            wk = tf.get_variable('wk_'+str(type),shape = [num_units,1])
            # alpha_k = tf.get_variable('alpha_k_'+str(type), shape = [1])
            alpha_k = - 0.1 # 论文里面将其设置为固定值
            bk = tf.get_variable('bk_'+str(type), shape = [1])

            # TODO 论文里面将分母加了1
            raw_lambda = alpha_k * (time_interval /(last_time + 1)) + tf.matmul(hidden_emb,wk) + bk
            # TODO
            # raw_lambda = tf.matmul(hidden_emb,wk)

            lambda_val = tf.nn.softplus(raw_lambda) # batch_size, 1
        return lambda_val



    def cal_target_intensity(self,hidden_emb,target_time, last_time,type_num):
        """

        :param hidden_emb: batch_size, num_units
        :param target_time: batch_size,
        :param last_time: batch_size,
        :param type_num: int
        :return:
        """
        lst = []
        target_time = tf.reshape(target_time, shape = (-1,1))
        last_time = tf.reshape(last_time, shape = (-1,1))
        time_interval = target_time - last_time
        for type in range(type_num):
            cur_intensity = self.cal_intensity(hidden_emb= hidden_emb,
                                               time_interval= time_interval,
                                               last_time= last_time,
                                               type = type)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis = 1) # batch_size, type_num
        return target_intensity


    def cal_sims_intensity(self,hidden_emb, sims_time, last_time, sims_len,type_num):
        """

        :param hidden_emb: batch_size, num_units
        :param sims_time: batch_size, sims_len
        :param last_time: batch_size,
        :param sims_len: int
        :param type_num: int
        :return:
        """
        num_units = hidden_emb.shape [-1]
        last_time = tf.reshape(last_time, [-1,1])
        last_time = tf.tile(last_time, [1, sims_len])
        sims_time = tf.reshape(sims_time,[-1,1]) # batch_size * sims_len , 1
        last_time = tf.reshape(last_time, [-1,1]) # batch_size * sims_len , 1
        time_interval = sims_time - last_time # batch_size * sims_len , 1

        hidden_emb = tf.tile(hidden_emb, [1, sims_len])
        hidden_emb = tf.reshape(hidden_emb,[-1,num_units]) # batch_size * sims_len , num_units

        lst = []
        for type in range(type_num):
            cur_intensity = self.cal_intensity(hidden_emb=hidden_emb,
                                               time_interval=time_interval,
                                               last_time=last_time,
                                               type = type) # batch_size * sims_len , 1
            lst.append(cur_intensity)
        sims_intensity = tf.concat(lst, axis = 1) # batch_size * sims_len, type_num
        sims_intensity = tf.reshape(sims_intensity,[-1,sims_len, type_num])
        return sims_intensity




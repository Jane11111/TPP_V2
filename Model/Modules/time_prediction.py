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
            b_time = tf.get_variable('b_time',shape = [1])

            times = tf.matmul(emb,W_time) + b_time
        return times

class hp_time_predictor():

    def __init__(self,f, type_num, max_seq_len,
                 type_lst, last_time,time_lst, seq_len):
        self.f = f
        self.type_num = type_num
        self.max_seq_len = max_seq_len
        self.last_time = last_time # batch_size,
        self.type_lst = type_lst # batch_size, max_seq_len
        self.time_lst = time_lst # batch_size, max_seq_len
        self.seq_len = seq_len # batch_size, for masking包含mask掉那一位

    def cal_total_intensity(self,target_time):
        """
        :param target_time: batch_size,
        :return: batch_size, 1   lambda(t)
        """
        target_time = tf.reshape(target_time, [-1,1])
        timenow_lst = target_time - self.time_lst
        lst = []
        for type in range(self.type_num):
            cur_intensity = self.f.cal_intensity(timenow_lst=timenow_lst, type_lst=self.type_lst,
                                               type_num=self.type_num, type_id=type,
                                               seq_len=self.seq_len, max_seq_len=self.max_seq_len)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis=1)  # batch_size, type_num
        lambda_t = tf.reduce_sum(target_intensity, axis= 1, keep_dims=True) # batch_size, 1
        return lambda_t


    def cal_total_origin_intensity(self,target_time):
        """
        原函数
        :param target_time: batch_size,
        :return: batch_size, 1   lambda(t)
        """
        target_time = tf.reshape(target_time, [-1, 1])
        timenow_lst = target_time - self.time_lst
        lst = []
        for type in range(self.type_num):
            cur_intensity = self.f.cal_origin_intensity(timenow_lst=timenow_lst, t_target=target_time,
                                                        type_lst=self.type_lst,type_id=type,
                                                        seq_len=self.seq_len, max_seq_len=self.max_seq_len)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis=1)  # batch_size, type_num
        lambda_t = tf.reduce_sum(target_intensity, axis= 1, keep_dims=True) # batch_size, 1
        return lambda_t

    def cal_inner_integral(self,t_last, t_target):
        """
        使用原函数计算内部积分值
        :param t_last:  batch_size,
        :param t_target:  batch_size,
        :return:
        """

        term1 = self.cal_total_origin_intensity(t_target)
        term2 = self.cal_total_origin_intensity(t_last)
        integral_intensity = term1-term2

        return integral_intensity # batch_size,1


    def predict_time(self,outer_sims_len):
        """
        在t_last至inf范围内计算 t*f(t)积分
        :param t_last: batch_size,
        :return:
        """

        """
        step1: 在t_last - inf上采样N 个样本（记做t_target）
        """
        t_last=tf.reshape(self.last_time,[-1,1])
        avg_inter_len = (t_last-0.)/(tf.to_float(tf.reshape(self.seq_len,[-1,1]))-1)

        sims_lst = []
        for i in range(outer_sims_len):
            sims_lst.append(t_last+avg_inter_len*(2**i))

        max_sims_lst = sims_lst[-1] # batch_size, 1
        """
        step2: 计算f(t) = lambda(t) * exp(-integral))
        """
        f_t_lst = []
        for cur_sims_lst in sims_lst:
            cur_sims_lst = tf.reshape(cur_sims_lst,shape=[-1,])
            term1 = self.cal_total_intensity(target_time= cur_sims_lst )
            term2 = self.cal_inner_integral(t_last = t_last, t_target= cur_sims_lst)
            term2 = tf.exp(-term2)
            f_t_lst.append(term1*term2)


        """
        step5: 使用t_last->max_t_target上的矩形面积作为积分近似
        """
        total_sims_lst = tf.concat(sims_lst,axis=1) # batch_size, outer_sims_len
        total_f_t_lst = tf.concat(f_t_lst, axis=1) # batch_size, outer_sims_len

        total_mul = total_sims_lst * total_f_t_lst # batch_size, outer_sims_len
        expectations = tf.reduce_mean(total_mul, axis=1, keep_dims=True)*max_sims_lst

        return expectations

class ihp_time_predictor():

    def __init__(self,f, type_num, max_seq_len,
                 type_lst, last_time,time_lst, seq_len):
        self.f = f
        self.type_num = type_num
        self.max_seq_len = max_seq_len
        self.last_time = last_time # batch_size,
        self.type_lst = type_lst # batch_size, max_seq_len
        self.time_lst = time_lst # batch_size, max_seq_len
        self.seq_len = seq_len # batch_size, for masking包含mask掉那一位

    def cal_total_intensity(self,target_time):
        """
        :param target_time: batch_size,
        :return: batch_size, 1   lambda(t)
        """
        target_time = tf.reshape(target_time,[-1,1])
        timenow_lst = target_time - self.time_lst
        lst = []
        for type in range(self.type_num):
            cur_intensity = self.f.cal_intensity(timenow_lst=timenow_lst, type_lst=self.type_lst,
                                               type_num=self.type_num, type_id=type,
                                               seq_len=self.seq_len, max_seq_len=self.max_seq_len)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis=1)  # batch_size, type_num
        lambda_t = tf.reduce_sum(target_intensity, axis= 1, keep_dims=True) # batch_size, 1
        return lambda_t




    def cal_inner_integral(self,t_last, t_target,inner_sims_len):
        """
        :param t_last:  batch_size,
        :param t_target:  batch_size,
        :return:
        """

        """
        step1: 在 t_last和t_target之间采样N=10个值
        """
        t_last = tf.reshape( t_last, [-1 ,1])
        t_target = tf.reshape(t_target, [-1,1])
        inter_len = (t_target-t_last)/inner_sims_len
        sims_lst = []
        for i in range(inner_sims_len):
            sims_lst.append(t_last+(i+1)*inter_len)

        """
        step2: 计算每一个采样的lambda(s)
        """
        sims_intensity = []
        for cur_sims_lst in sims_lst:
            cur_sims_intensity = self.cal_total_intensity(target_time=cur_sims_lst)  # f(s) batch_size,1
            sims_intensity.append(cur_sims_intensity)

        all_sims_intensity = tf.concat(sims_intensity, axis=1)  # batch_size, inner_sims_len

        """
        step3: 使用矩形面积近似积分
        """
        sum_sims_intensity = tf.reduce_sum(all_sims_intensity, axis=1, keep_dims=True)  # batch_size,1
        integral_intensity = (sum_sims_intensity / inner_sims_len) * (t_target - t_last)  # batch_size, 1

        return integral_intensity


    def predict_time(self,inner_sims_len,outer_sims_len):
        """
        在t_last至inf范围内计算 t*f(t)积分
        :param t_last: batch_size,
        :return:
        """

        """
        step1: 在t_last - inf上采样N 个样本（记做t_target）
        """
        t_last=tf.reshape(self.last_time,[-1,1])
        avg_inter_len = (t_last-0.)/(tf.to_float(self.seq_len)-1)
        if avg_inter_len ==0:
            avg_inter_len  = 1
        sims_lst = []
        for i in range(outer_sims_len):
            sims_lst.append(t_last+avg_inter_len*(2**i))

        max_sims_lst = sims_lst[-1] # batch_size, 1
        """
        step2: 计算f(t) = lambda(t) * exp(-integral))
        """
        f_t_lst = []
        for cur_sims_lst in sims_lst:
            cur_sims_lst = tf.reshape(cur_sims_lst,shape=[-1,])
            term1 = self.cal_total_intensity(target_time=cur_sims_lst)
            term2 = self.cal_inner_integral(t_last = t_last, t_target=cur_sims_lst,inner_sims_len=inner_sims_len)
            term2 = tf.exp(-term2)
            f_t_lst.append(term1*term2)


        """
        step5: 使用t_last->max_t_target上的矩形面积作为积分近似
        """
        total_sims_lst = tf.concat(sims_lst,axis=1) # batch_size, outer_sims_len
        total_f_t_lst = tf.concat(f_t_lst, axis=1) # batch_size, outer_sims_len

        total_mul = total_sims_lst * total_f_t_lst # batch_size, outer_sims_len
        expectations = tf.reduce_mean(total_mul, axis=1, keep_dims=True)*max_sims_lst

        return expectations





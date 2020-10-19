# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 11:02
# @Author  : zxl
# @FileName: intensity_calculation.py

import tensorflow.compat.v1 as tf

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

class type_intensity():
    """
    使用type table 计算强度
    """

    def __init__(self,type_predictor, num_units,W,type_num):
        """

        :param W: type_num, num_units
        """
        self.type_predictor= type_predictor
        self.num_units = num_units
        self.type_num = type_num
        self.W = W


    def cal_intensity(self,emb):

        """

        :param emb: batch_size, num_units
        :return:  batch_size, type_num
        """

        # intensity = tf.nn.softplus(tf.matmul(emb, self.W, transpose_b= True))
        # return intensity
        intensity = self.type_predictor.predict_type(emb=emb,
                                         num_units=self.num_units,
                                         type_num=self.type_num,
                                         type_table = self.W)
        intensity = tf.nn.softplus(intensity)
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

class e_intensity():


    def __init__(self,type_predictor, num_units,W,type_num):
        """

        :param W: type_num, num_units
        """
        self.type_predictor= type_predictor
        self.num_units = num_units
        self.type_num = type_num
        self.W = W


    def cal_intensity(self,emb):

        """

        :param emb: batch_size, num_units
        :return:  batch_size, type_num
        """

        intensity = tf.nn.softplus(tf.matmul(emb, self.W, transpose_b= True))
        return intensity
        # intensity = self.type_predictor.predict_type(emb=emb,
        #                                  num_units=self.num_units,
        #                                  type_num=self.type_num,
        #                                  type_table = self.W)
        # intensity = tf.nn.softmax(intensity)
        # return intensity

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

class w_intensity():


    def __init__(self,  num_units ):
        """

        :param W: type_num, num_units
        """
        self.num_units = num_units


    def cal_intensity(self,emb):

        """

        :param emb: batch_size, num_units
        :return:  batch_size, type_num
        """

        intensity = tf.layers.dense(emb,units=1,activation=tf.nn.softplus)
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

            # TODO 应该在这里将last_time + 1不然计算的interval 会错误
            raw_lambda = alpha_k * (time_interval /(last_time + 1 )) + tf.matmul(hidden_emb,wk) + bk
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

class sahp_intensity_calculation():

    def __init__(self,mu,eta,gamma):
        self.mu = mu
        self.eta = eta
        self.gamma = gamma


    def cal_intensity(self,mu,eta,gamma,time_interval,type):
        """
        :param time_interval: N,1
        :param type: int
        :return:
        """
        with tf.variable_scope('intensity_calculation'+'_'+str(type)):
            raw_lambda = mu +(eta-mu) * tf.exp(-gamma * time_interval)

            lambda_val = tf.nn.softplus(raw_lambda) # batch_size, 1
        return lambda_val



    def cal_target_intensity(self,target_time, last_time,type_num):
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
            cur_intensity = self.cal_intensity(mu = self.mu,eta=self.eta, gamma=self.gamma,time_interval= time_interval,
                                               type = type)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis = 1) # batch_size, type_num
        return target_intensity


    def cal_sims_intensity(self,sims_time, last_time, sims_len,type_num):
        """

        :param sims_time: batch_size, sims_len
        :param last_time: batch_size,
        :param sims_len: int
        :param type_num: int
        :return:
        """
        last_time = tf.reshape(last_time, [-1,1])
        last_time = tf.tile(last_time, [1, sims_len])
        sims_time = tf.reshape(sims_time,[-1,1]) # batch_size * sims_len , 1
        last_time = tf.reshape(last_time, [-1,1]) # batch_size * sims_len , 1
        time_interval = sims_time - last_time # batch_size * sims_len , 1

        mu = tf.reshape(tf.tile(self.mu,[1,sims_len]),[-1,1])
        eta = tf.reshape(tf.tile(self.eta,[1,sims_len]),[-1,1])
        gamma = tf.reshape(tf.tile(self.gamma,[1,sims_len]),[-1,1])

        lst = []
        for type in range(type_num):
            cur_intensity = self.cal_intensity(mu=mu,eta=eta,gamma=gamma,time_interval=time_interval,
                                               type = type) # batch_size * sims_len , 1
            lst.append(cur_intensity)
        sims_intensity = tf.concat(lst, axis = 1) # batch_size * sims_len, type_num
        sims_intensity = tf.reshape(sims_intensity,[-1,sims_len, type_num])
        return sims_intensity



class nhp_intensity_calculation():

    def __init__(self):
        pass

    def cal_intensity(self,hidden_emb,type):
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


            raw_lambda =  tf.matmul(hidden_emb,wk) + bk


            lambda_val = tf.nn.softplus(raw_lambda) # batch_size, 1
        return lambda_val



    def cal_target_intensity(self,hidden_emb,type_num):
        """

        :param hidden_emb: batch_size, num_units
        :param type_num: int
        :return:
        """
        lst = []
        for type in range(type_num):
            cur_intensity = self.cal_intensity(hidden_emb= hidden_emb,
                                               type = type)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis = 1) # batch_size, type_num
        return target_intensity


    def cal_sims_intensity(self,hidden_emb,sims_len,type_num):
        """
        :param hidden_emb: batch_size,seq_len, num_units
        :param sims_len: int
        :param type_num: int
        :return:
        """
        num_units = hidden_emb.shape [-1]

        hidden_emb = tf.reshape(hidden_emb,[-1,num_units]) # batch_size * sims_len , num_units

        lst = []
        for type in range(type_num):
            cur_intensity = self.cal_intensity(hidden_emb=hidden_emb,
                                               type = type) # batch_size * sims_len , 1
            lst.append(cur_intensity)
        sims_intensity = tf.concat(lst, axis = 1) # batch_size * sims_len, type_num
        sims_intensity = tf.reshape(sims_intensity,[-1,sims_len, type_num])
        return sims_intensity

class rmtpp_density_calculation():
    def __init__(self):
        pass

    def cal_density(self,hidden_emb,time_interval ):
        """

        :param hidden_emb: N, num_units
        :param time_interval: N,1
        :param last_time: N,1
        :param type: int
        :return: batch_size,1
        """
        num_units = hidden_emb.shape[1]
        dtype = hidden_emb.dtype
        with tf.variable_scope('density_calculation'):
            vt = tf.get_variable("vt", shape=[num_units, 1], dtype=dtype)  # input_size, num_units
            wt = tf.get_variable("wt", shape=[1, 1], dtype=dtype)  # input_size, num_units
            bt = tf.get_variable("bt", shape=[1,1], dtype=dtype)  # input_size, num_units
            small_num = tf.constant(1,dtype=dtype,shape=[1,1])
            small_num2 = tf.ones_like(time_interval) * 5

        tmp = tf.cast(tf.to_int32(wt > 0.) + tf.to_int32(wt < 0.), dtype=tf.bool)
        wt = tf.where(tmp, wt, small_num)
        intensity_val = tf.matmul(hidden_emb,vt) + wt * time_interval + bt
        f_val = intensity_val + 1/(wt) * tf.exp(tf.matmul(hidden_emb,vt) + bt) - 1/(wt) * tf.exp(intensity_val)
        # 截断
        f_val = tf.where(f_val<5,f_val,small_num2)
        f_val = tf.exp(f_val)


        return f_val



    def cal_target_density(self,hidden_emb,target_time, last_time):
        """

        :param hidden_emb: batch_size, num_units
        :param target_time: batch_size,
        :param last_time: batch_size,
        :param type_num: int
        :return:
        """
        interval = tf.expand_dims(target_time-last_time, axis=1)

        f_val = self.cal_density(hidden_emb=hidden_emb,time_interval=interval)

        return f_val



class hp_intensity_calculation():

    def __init__(self):
        pass

    def cal_intensity(self,timenow_lst,type_lst,type_num,type_id,seq_len,max_seq_len):
        """
        :param timenow_lst: N, max_seq_len
        :param type_lst: N, max_seq_len
        :param type_num:  int 事件类型总数
        :param type_id: 当前事件id
        :param seq_len: N,
        :param max_seq_len: int
        :return:
        """

        dtype=timenow_lst.dtype

        with tf.variable_scope('single_type_intensity_calculation'):
            self.mu =  tf.get_variable('mu', shape=(type_num+1,), dtype=dtype)
            self.alpha =  tf.get_variable('alpha', shape=(type_num+1, type_num+1), dtype=dtype)
            self.delta =  tf.get_variable('delta', shape=(type_num+1, type_num+1), dtype=dtype)


        row_idx = tf.reshape(type_lst, [-1, 1])  # N*seq_len, 1
        col_idx =  tf.ones_like(row_idx) * type_id
        idx = tf.concat([row_idx, col_idx], axis=1)
        # TODO 如何限制为正
        cur_mu = tf.nn.relu(self.mu[type_id])
        cur_alpha =tf.nn.relu(tf.reshape(tf.gather_nd(self.alpha, idx), shape=(-1, max_seq_len))) # N, max_seq_len
        cur_delta = tf.nn.relu(tf.reshape(tf.gather_nd(self.delta,idx), shape=(-1,max_seq_len)))

        term2 = cur_alpha * tf.exp(-(cur_delta) * timenow_lst) # N, max_seq_len

        masks = tf.sequence_mask(seq_len-1,maxlen=max_seq_len)# N, max_seq_len
        term2 = term2 * tf.to_float(masks)
        term2 = tf.reduce_sum(term2, axis=1, keep_dims=True)

        lambda_val = cur_mu + term2

        return lambda_val # batch_size, 1

    def cal_origin_intensity(self,timenow_lst,t_target,type_lst,type_id,seq_len,max_seq_len):
        """
        强度函数的原函数在t=t_target时的值
        :param timenow_lst: N, max_seq_len
        :param t_target: N,
        :param type_lst: N, max_seq_len
        :param type_num:  int 事件类型总数
        :param type_id: 当前事件id
        :param seq_len: N,
        :param max_seq_len: int
        :return:
        """
        masks = tf.sequence_mask(seq_len-1,maxlen=max_seq_len)# N, max_seq_len


        row_idx = tf.reshape(type_lst, [-1, 1]) # N*seq_len, 1
        col_idx = tf.ones_like(row_idx) * type_id
        idx = tf.concat([row_idx, col_idx], axis=1)
        # TODO 如何限制为正
        cur_mu =  tf.nn.relu(self.mu[type_id])
        cur_alpha =  tf.nn.relu(tf.reshape(tf.gather_nd(self.alpha, idx), shape=(-1, max_seq_len)) )# N, max_seq_len
        cur_delta =  tf.nn.relu(tf.reshape(tf.gather_nd(self.delta,idx), shape=(-1,max_seq_len)))

        cur_alpha *= tf.to_float(masks)
        cur_delta *= tf.to_float(masks)

        term2 = (cur_alpha/(-(cur_delta+0.001))) * tf.exp(-cur_delta * timenow_lst) # N, max_seq_len

        term2 = term2 * tf.to_float(masks)
        term2 = tf.reduce_sum(term2, axis=1, keep_dims=True)

        t_target = tf.reshape(t_target, [-1,1])
        lambda_val = cur_mu * t_target + term2

        return lambda_val # batch_size, 1


    def cal_target_intensity(self,timenow_lst, type_lst,type_num,seq_len, max_seq_len):
        """

        :param timenow_lst: batch_size, max_seq_len
        :param type_lst:  batch_size, max_seq_len
        :param type_num: int
        :param seq_len: batch_size,
        :param max_seq_len: int
        :return:
        """
        lst = []
        for type in range(type_num):
            cur_intensity = self.cal_intensity(timenow_lst=timenow_lst,type_lst=type_lst,
                                               type_num=type_num, type_id = type,
                                               seq_len=seq_len, max_seq_len=max_seq_len)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis = 1) # batch_size, type_num
        return target_intensity



    def cal_total_origin_intensity(self,target_time, time_lst,type_lst, type_num, seq_len,max_seq_len):
        """
        原函数
        :param target_time: batch_size,
        :return: batch_size, 1   lambda(t)
        """
        target_time = tf.reshape(target_time, [-1, 1])
        timenow_lst = target_time - time_lst
        lst = []
        for type in range( type_num):
            cur_intensity = self.cal_origin_intensity(timenow_lst=timenow_lst, t_target=target_time,
                                                        type_lst= type_lst,type_id=type,
                                                        seq_len= seq_len, max_seq_len= max_seq_len)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis=1)  # batch_size, type_num
        lambda_t = tf.reduce_sum(target_intensity, axis= 1, keep_dims=True) # batch_size, 1
        return lambda_t

    def cal_integral_intensity(self, t_last,t_target, time_lst, type_lst, type_num, seq_len, max_seq_len):
        term2 = self.cal_total_origin_intensity(target_time = t_target,
                                                time_lst = time_lst,
                                                type_lst = type_lst,
                                                type_num = type_num,
                                                seq_len = seq_len,
                                                max_seq_len = max_seq_len)
        term1 = self.cal_total_origin_intensity(target_time = t_last,
                                                time_lst = time_lst,
                                                type_lst = type_lst,
                                                type_num = type_num,
                                                seq_len=seq_len , # 为什么这一位对预测时间的影响这么大
                                                max_seq_len=max_seq_len)
        return term2-term1

    #
    def cal_sims_intensity(self,sims_timenow_lst,type_lst,type_num,seq_len,max_seq_len,sims_len):
        """

        :param sims_timenow_lst: batch_size, sims_len, max_seq_len
        :param type_lst: batch_size, max_seq_len
        :param type_num: int
        :param seq_len: batch_size,
        :param max_seq_len: int
        :param sims_len: int
        :return:
        """
        sims_timenow_lst = tf.split(sims_timenow_lst, num_or_size_splits=sims_len, axis=1)
        lst = []
        for sims_timenow in sims_timenow_lst:
            sims_timenow = tf.reshape(sims_timenow,shape=(-1,max_seq_len))
            sub_lst = []
            for type in range(type_num):
                cur_intensity = self.cal_intensity(timenow_lst=sims_timenow,type_lst=type_lst,
                                                   type_num=type_num, type_id = type,
                                                   seq_len = seq_len, max_seq_len = max_seq_len) # batch_size ,1
                sub_lst.append(cur_intensity)
            sub_intensity = tf.concat(sub_lst,axis=1) # batch_size, type_num
            lst.append(sub_intensity)
        sims_intensity = tf.concat(lst, axis=1) # batch_size, type_num*sims_len
        sims_intensity = tf.reshape(sims_intensity,shape=(-1,sims_len,type_num))

        return sims_intensity # bath_size, sims_len, type_num




class ihp_intensity_calculation():

    def __init__(self):
        pass


    def cal_intensity(self,timenow_lst,type_lst,type_num,type_id,seq_len,max_seq_len):
        """
        :param timenow_lst: N, max_seq_len
        :param type_lst: N, max_seq_len
        :param type_num:  int 事件类型总数
        :param type_id: 当前事件id
        :param seq_len: N,
        :param max_seq_len: int
        :return:
        """
        dtype=timenow_lst.dtype
        with tf.variable_scope('single_type_intensity_calculation'):
            self.mu = tf.get_variable('mu', shape=(type_num+1,), dtype=dtype)
            self.s = tf.get_variable('s', shape=(type_num+1,), dtype=dtype)
            self.alpha = tf.get_variable('alpha', shape=(type_num+1, type_num+1), dtype=dtype)
            self.delta = tf.get_variable('delta', shape=(type_num+1, type_num+1), dtype=dtype)

        row_idx = tf.reshape(type_lst, [-1, 1]) # N*seq_len, 1
        col_idx = tf.ones_like(row_idx) * type_id
        idx = tf.concat([row_idx, col_idx], axis=1)

        cur_mu = self.mu[type_id]
        cur_s = tf.nn.relu(self.s[type_id])+1e-9 # the intensity must be positive
        cur_alpha = tf.reshape(tf.gather_nd(self.alpha, idx), shape=(-1, max_seq_len)) # N, max_seq_len
        cur_delta = tf.reshape(tf.gather_nd(self.delta,idx), shape=(-1,max_seq_len))

        term2 =   cur_alpha * tf.exp(-cur_delta * timenow_lst) # N, max_seq_len

        masks = tf.sequence_mask(seq_len-1,maxlen=max_seq_len)# N, max_seq_len
        term2 = term2 * tf.to_float(masks)
        term2 = tf.reduce_sum(term2, axis=1, keep_dims=True)

        lambda_bar = cur_mu + term2

        # TODO scaled softplus
        lambda_val = cur_s*tf.nn.softplus(lambda_bar/cur_s)

        return lambda_val # batch_size, 1


    def cal_target_intensity(self,timenow_lst, type_lst,type_num,seq_len, max_seq_len):
        """

        :param timenow_lst: batch_size, max_seq_len
        :param type_lst:  batch_size, max_seq_len
        :param type_num: int
        :param seq_len: batch_size,
        :param max_seq_len: int
        :return:
        """
        lst = []
        for type in range(type_num):
            cur_intensity = self.cal_intensity(timenow_lst=timenow_lst,type_lst=type_lst,
                                               type_num=type_num, type_id = type,
                                               seq_len=seq_len, max_seq_len=max_seq_len)
            lst.append(cur_intensity)
        target_intensity = tf.concat(lst, axis = 1) # batch_size, type_num
        return target_intensity


    def cal_sims_intensity(self,sims_timenow_lst,type_lst,type_num,seq_len,max_seq_len,sims_len):
        """

        :param sims_timenow_lst: batch_size, sims_len, max_seq_len
        :param type_lst: batch_size, max_seq_len
        :param type_num: int
        :param seq_len: batch_size,
        :param max_seq_len: int
        :param sims_len: int
        :return:
        """
        sims_timenow_lst = tf.split(sims_timenow_lst, num_or_size_splits=sims_len, axis=1)
        lst = []
        for sims_timenow in sims_timenow_lst:
            sims_timenow = tf.reshape(sims_timenow,shape=(-1,max_seq_len))
            sub_lst = []
            for type in range(type_num):
                cur_intensity = self.cal_intensity(timenow_lst=sims_timenow,type_lst=type_lst,
                                                   type_num=type_num, type_id = type,
                                                   seq_len = seq_len, max_seq_len = max_seq_len) # batch_size ,1
                sub_lst.append(cur_intensity)
            sub_intensity = tf.concat(sub_lst,axis=1) # batch_size, type_num
            lst.append(sub_intensity)
        sims_intensity = tf.concat(lst, axis=1) # batch_size, type_num*sims_len
        sims_intensity = tf.reshape(sims_intensity,shape=(-1,sims_len,type_num))

        return sims_intensity # bath_size, sims_len, type_num
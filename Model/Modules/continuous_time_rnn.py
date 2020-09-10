# -*- coding: utf-8 -*-
# @Time    : 2020/9/9 14:28
# @Author  : zxl
# @FileName: continuous_time_rnn.py
import logging
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops import math_ops,array_ops,variable_scope
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn

class ContinuousLSTM():


    def build_ctsm_cell(self,hidden_units):
        cell  = ContinuousLSTMCell(hidden_units)
        return MultiRNNCell([cell])

    def ctsm_net(self, hidden_units, input_data, input_length, scope='ctsm'):

        cell = self.build_ctsm_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs, _ = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32,scope= scope)
        return outputs


class ContinuousLSTMCell(RNNCell):

    def __init__(self,num_units, forget_bias=1.0,activation=None,reuse=None):
        super(ContinuousLSTMCell,self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return  5 * self._num_units

    @property
    def output_size(self):
        return  5* self._num_units


    def call(self,inputs, state):
        sigmoid = math_ops.sigmoid
        dtype = inputs.dtype

        k = inputs[:,:-2]
        time_last = tf.expand_dims(inputs[:,-1],axis=1)# 当前时间与上次时间的间隔

        inputs = inputs[:,:-2]
        input_size = inputs.get_shape().with_rank(2)[1]

        o_i,c_i,c_i_bar,delta_i,_ = array_ops.split(value=state,num_or_size_splits=5,axis=1)

        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):

                Wi = variable_scope.get_variable(
                    "Wi", shape=[input_size,self._num_units], dtype=dtype)# input_size, num_units
                Ui = variable_scope.get_variable(
                    "Ui", shape=[self._num_units,self._num_units],dtype=dtype
                )# num_units, num_units
                di = variable_scope.get_variable(
                    "di",shape=[self._num_units],dtype=dtype
                )# num_units
                Wi_bar = variable_scope.get_variable(
                    "Wi_bar", shape=[input_size, self._num_units], dtype=dtype)  # input_size, num_units
                Ui_bar = variable_scope.get_variable(
                    "Ui_bar", shape=[self._num_units, self._num_units], dtype=dtype
                )  # num_units, num_units
                di_bar = variable_scope.get_variable(
                    "di_bar", shape=[self._num_units], dtype=dtype
                )  # num_units
                Wf = variable_scope.get_variable(
                    "Wf", shape=[input_size, self._num_units], dtype=dtype)  # input_size, num_units
                Uf = variable_scope.get_variable(
                    "Uf", shape=[self._num_units, self._num_units], dtype=dtype
                )  # num_units, num_units
                df = variable_scope.get_variable(
                    "df", shape=[self._num_units], dtype=dtype
                )  # num_units
                Wf_bar = variable_scope.get_variable(
                    "Wf_next_bar", shape=[input_size, self._num_units], dtype=dtype)  # input_size, num_units
                Uf_bar = variable_scope.get_variable(
                    "Uf_next_bar", shape=[self._num_units, self._num_units], dtype=dtype
                )  # num_units, num_units
                df_bar = variable_scope.get_variable(
                    "df_next_bar", shape=[self._num_units], dtype=dtype
                )  # num_units
                Wz = variable_scope.get_variable(
                    "Wz", shape=[input_size, self._num_units], dtype=dtype)  # input_size, num_units
                Uz = variable_scope.get_variable(
                    "Uz", shape=[self._num_units, self._num_units], dtype=dtype
                )  # num_units, num_units
                dz = variable_scope.get_variable(
                    "dz", shape=[self._num_units], dtype=dtype
                )  # num_units
                Wo = variable_scope.get_variable(
                    "Wo", shape=[input_size, self._num_units], dtype=dtype)  # input_size, num_units
                Uo = variable_scope.get_variable(
                    "Uo", shape=[self._num_units, self._num_units], dtype=dtype
                )  # num_units, num_units
                do = variable_scope.get_variable(
                    "do", shape=[self._num_units], dtype=dtype
                )  # num_units
                Wd = variable_scope.get_variable(
                    "Wd", shape=[input_size, self._num_units], dtype=dtype)  # input_size, num_units
                Ud = variable_scope.get_variable(
                    "Ud", shape=[self._num_units, self._num_units], dtype=dtype
                )  # num_units, num_units
                dd = variable_scope.get_variable(
                    "dd", shape=[self._num_units], dtype=dtype
                )  # num_units
        # h(ti)

        c_ti = c_i_bar + (c_i-c_i_bar)* math_ops.exp(-delta_i * (time_last)) # batch_size,  num_units

        h_ti = o_i * (2* sigmoid(2* c_ti) - 1) # batch_size, num_units

        i_next = sigmoid(math_ops.matmul(k,Wi) + math_ops.matmul(h_ti, Ui) + di)
        f_next = sigmoid(math_ops.matmul(k,Wf) + math_ops.matmul(h_ti,Uf) + df)
        z_next = 2* sigmoid(math_ops.matmul(k,Wz) + math_ops.matmul(h_ti, Uz)+dz) - 1
        o_next = sigmoid(math_ops.matmul(k, Wo) + math_ops.matmul(h_ti, Uo) + do)

        i_next_bar = sigmoid(math_ops.matmul(k, Wi_bar) + math_ops.matmul(h_ti, Ui_bar) + di_bar)
        f_next_bar = sigmoid(math_ops.matmul(k, Wf_bar) + math_ops.matmul(h_ti, Uf_bar) + df_bar)

        c_next = f_next * c_ti + i_next * z_next
        c_next_bar = f_next_bar * c_i_bar + i_next_bar * z_next
        delta_next = tf.nn.softplus(math_ops.matmul(k,Wd) + math_ops.matmul(h_ti,Ud)+dd)

        next_state = array_ops.concat([o_next,c_next,c_next_bar,delta_next,h_ti],axis=1)

        return   next_state,next_state
# -*- coding: utf-8 -*-
# @Time    : 2020/9/11 10:25
# @Author  : zxl
# @FileName: rmtpp_rnn.py

import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import RNNCell
from tensorflow.python.ops import math_ops,array_ops,variable_scope
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn

class Rmtpp_RNN():


    def build_ctsm_cell(self,hidden_units):
        cell  = RmtppRNNCell(hidden_units)
        return MultiRNNCell([cell])

    def rmtpp_net(self, hidden_units, input_data, input_length, scope='ctsm'):

        cell = self.build_ctsm_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs,state = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32,scope= scope)
        return outputs,state


class RmtppRNNCell(RNNCell):

    def __init__(self,num_units, forget_bias=1.0,activation=None,reuse=None):
        super(RmtppRNNCell,self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return  self._num_units

    @property
    def output_size(self):
        return self._num_units


    def call(self,inputs, state):
        sigmoid = math_ops.sigmoid
        dtype = inputs.dtype

        y_j = inputs[:,:-1]
        t_j = tf.expand_dims(inputs[:,-1],axis=1)# 当前时间与上次时间的间隔

        inputs = inputs[:,:-1]
        input_size = inputs.get_shape().with_rank(2)[1]

        h_j_minus = state

        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):

                Wy = variable_scope.get_variable(
                    "Wy", shape=[input_size,self._num_units], dtype=dtype)# input_size, num_units
                Wt = variable_scope.get_variable(
                    "Wt", shape = [1,self._num_units],dtype=dtype
                )
                Wh = variable_scope.get_variable(
                    "Wh", shape=[self._num_units,self._num_units],dtype=dtype
                )# num_units, num_units
                bh = variable_scope.get_variable(
                    "bh",shape=[self._num_units],dtype=dtype
                )# num_units


        h_j = math_ops.matmul(y_j,Wy) + math_ops.matmul(t_j,Wt) + math_ops.matmul(h_j_minus,Wh) + bh
        h_j = tf.nn.relu(h_j)

        return   h_j,h_j
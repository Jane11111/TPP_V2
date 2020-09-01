# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 19:26
# @Author  : zxl
# @FileName: transformer_encoder.py

import tensorflow as tf

class transformer_encoder():


    def __init__(self):
        pass


    def normalize(self,inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
          `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def stack_multihead_self_attention(self,stack_num,type_enc,time_enc,M,Mk,Mv,Mi,L,N,head_num,dropout_rate,reuse= None,scope = 'stack_multihead_self_attention'):

        X = type_enc

        with tf.variable_scope(scope, reuse= reuse):

            for i in range(stack_num):
                X += time_enc
                X = self.multihead_self_attention(X = X,
                                                  M = M,
                                                  Mk = Mk,
                                                  Mv = Mv,
                                                  L = L,
                                                  N = N,
                                                  head_num= head_num,
                                                  dropout_rate = dropout_rate,
                                                  reuse=reuse,
                                                  scope = 'block_'+str(i)+'_multihead_self_attention')
                # TODO 需要增加全连接层


        return X # batch_size, L, M


    def positionwise_feedforward(self,X,M,Mi,droppout_rate, reuse=None, scope = 'positionwise_feedforward'):

        """

        :param X: batch_size, seq_len, M
        :param M: int
        :param Mi: int
        :param droppout:
        :param reuse:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope,reuse=reuse):
            residual = X

            X = tf.layers.dense(inputs=X,
                                 units=Mi,
                                 activation=tf.nn.relu) # TODO gelu??
            X = tf.layers.dropout(inputs=X,
                                  rate=droppout_rate)
            X = tf.layers.dense(inputs = X,
                                units = M,
                                activation=tf.nn.relu) #TODO activation??
            X = tf.layers.dropout(inputs=X,
                                  rate = droppout_rate)
            X += residual

            X = self.normalize(inputs=X) # TODO 这个normalize是不是源代码里面的？
        return X

    def multihead_self_attention(self,X,M,Mk,Mv,L,N,head_num,dropout_rate,reuse=None,scope = 'multihead_self_attention'):
        """

        :param X: batch_size, shape = (batch_size,L, M)
        :param M: M
        :param Mk: Mk
        :param Mv: Mv
        :param head_num: int
        :param reuse:
        :param scope:
        :return:
        """

        with tf.variable_scope(scope,reuse=reuse):
            S = []
            for cur_head in range(head_num):
                Wq = tf.get_variable(str(cur_head)+'_Wq',shape = (M,Mk))
                Wk = tf.get_variable(str(cur_head)+'_Wk',shape = (M,Mk))
                Wv = tf.get_variable(str(cur_head)+'_Wv',shape = (M,Mv))

                Q = tf.reshape(tf.matmul(tf.reshape(X,[-1,M]),Wq),shape = [-1,L,Mk])# batch_size, L, Mk
                K = tf.reshape(tf.matmul(tf.reshape(X,[-1,M]),Wk),shape = [-1,L,Mk])
                V = tf.reshape(tf.matmul(tf.reshape(X,[-1,M]),Wv),shape = [-1,L,Mv])

                att = tf.matmul(Q,K,transpose_b=True)/(Mk**0.5) # batch_size, L, L


                # TODO mask the attention value
                masked_idx = tf.range(start = 1, limit = L+1,delta = 1) # L, L
                masks = tf.expand_dims(tf.sequence_mask(masked_idx,L),axis = 0)
                masks = tf.tile(masks,[N,1,1])
                paddings = tf.ones_like(att) * (-2 ** 32 +1)
                masked_att = tf.where(masks,att,paddings)

                masked_att = tf.nn.softmax(masked_att)

                # Dropouts
                masked_att = tf.layers.dropout(masked_att,rate = dropout_rate)

                # Weighted sum
                Si = tf.matmul(masked_att,V) # batch_size, L, Mv
                S.append(Si)

            # Restore shape
            S = tf.concat(S,axis=2) # batch_size,L, head_num * Mv

            Wo = tf.get_variable('Wo',shape = (head_num * Mv, M))
            S = tf.reshape(tf.matmul(tf.reshape(S,[-1,head_num * Mv]),Wo), shape = [-1,L, M]) # batch_size, L, M

        # Residual connection
        S += X

        # normalize
        S = self.normalize(inputs = S)

        return S


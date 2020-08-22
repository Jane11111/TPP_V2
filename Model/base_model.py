# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 09:51
# @Author  : zxl
# @FileName: base_model.py



import os
import time
from util.model_log import create_log
from Model.Modules.net_utils import *


class base_model(object):


    def __init__(self,FLAGS,Embedding):
        self.FLAGS = FLAGS
        self.learning_rate = tf.placeholder(tf.float64,[],name = 'learning_rate') #learning_rate都设置成输入了

        if self.FLAGS.checkpoint_path_dir != None:
            self.checkpoint_path_dir = self.FLAGS.checkpoint_path_dir
        else:
            self.checkpoint_path_dir = 'data/check_point/' + self.FLAGS.model_name
        if not os.path.exists(self.checkpoint_path_dir):
            os.makedirs(self.checkpoint_path_dir)

        self.init_optimizer()
        self.embedding = Embedding
        log_ins = create_log()
        self.logger = log_ins.logger

    def init_variables(self, sess):

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


    def init_optimizer(self):

        if self.FLAGS.optimizer == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate = self.learning_rate)
        elif self.FLAGS.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        elif self.FLAGS.optimizer == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate= self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate= self.learning_rate)

    def build_model(self):
        pass


    def save(self, sess, global_step = None, path = None, variable_lst = None):

        if path == None:
            path = self.checkpoint_path_dir
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, 'model.ckpt')

        saver = tf.train.Saver(var_list = variable_lst)
        save_path = saver.save(sess, save_path = path, global_step=global_step)


        self.logger.info('model saved at %s' % save_path)

    def restore(self, sess, path, variable_lst = None, graph_path = None):
        if graph_path != None:
            saver = tf.train.import_meta_graph(graph_path)
        else:
            saver = tf.train.Saver(var_list = variable_lst)

        saver.restore(sess, tf.train.latest_checkpoint(path))
        self.logger.info('model restored from %s' %path)


    def train(self, sess, batch_data, learning_rate):

        input_dic = self.embedding.make_feed_dic(batch_data = batch_data)

        input_dic[self.learning_rate] = learning_rate
        input_dic[self.now_batch_size] = len(batch_data)

        output_feed = [self.loss, self.l2_norm,self.merged]

        outputs = sess.run(output_feed, input_dic)

        return outputs

    def metrics_likelihood(self,sess,batch_data):

        output_feed = [tf.nn.softmax(self.lambda_prob),self.labels]
        input_dic = self.embedding.make_feed_dic(batch_data = batch_data)
        input_dic[self.now_batch_size] = len(batch_data)
        outputs = sess.run(output_feed,input_dic)
        return outputs

    def summary(self):

        self.merged = tf.summary.merge_all()

        time_array = time.localtime(time.time())
        time_str = time.strftime('%Y-%m-%d--%H-%M-%S', time_array)
        model_name = self.FLAGS.model_name
        filename = 'data/tensorboard_result/'

        filename = filename + model_name + '_' + time_str

        self.train_writer = tf.summary.FileWriter(filename + '/tensorboard_train')
        self.eval_writer = tf.summary.FileWriter(filename + '/tensorboard_eval')


    def cal_gradient(self,trainable_params):

        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _  = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params))
        self.summary()



    def output(self):

        with tf.name_scope('likelihood_loss'):

            self.l2_norm = tf.add_n([
                tf.nn.l2_loss(self.type_lst_embedding)
            ])
            regulation_rate = self.FLAGS.regulation_rate
            one_hot_type = tf.one_hot(
                self.target_type, depth = self.FLAGS.type_num, dtype = tf.float32
            )
            self.labels = tf.reshape(one_hot_type,[-1,self.FLAGS.type_num])



            # 将预测的值换成0、1
            # one = tf.ones_like(self.lambda_prob)
            # zero = tf.zeros_like(self.lambda_prob)
            #
            # max_lambda = tf.reshape(tf.reduce_max(self.lambda_prob),[-1,1])
            # resize_lambda = self.lambda_prob/max_lambda
            # sparse_lambda = tf.where(resize_lambda <1, x = zero, y=one)


            # cross entropy loss
            # cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.labels,
            #                                                                 logits = self.lambda_prob)
            #
            # self.cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
            # self.loss = self.cross_entropy_loss

            # pairwise loss
            # pairwise_loss = tf.losses.mean_pairwise_squared_error(labels = self.labels,
            #                                                       predictions=tf.nn.softmax(self.lambda_prob))
            # self.loss = tf.reduce_mean(pairwise_loss)

            # 自定义loss
            log_probs = tf.nn.log_softmax(self.lambda_prob)

            self.loss_origin = -tf.reduce_sum(log_probs * self.labels, axis=[-1])
            self.loss = regulation_rate * self.l2_norm + tf.reduce_mean(self.loss_origin)

            tf.summary.scalar('l2_norm', self.l2_norm)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('loss', self.loss)
            # tf.summary.scalar('cross entropy loss', self.cross_entropy_loss)
        self.cal_gradient(tf.trainable_variables())

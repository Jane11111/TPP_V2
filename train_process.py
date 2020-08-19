# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 15:03
# @Author  : zxl
# @FileName: train_process.py



import time
import random
import numpy as np
import tensorflow as tf

from Embedding.Type_Embedding import Type_embedding
from util.model_log import create_log
from DataHandle.get_input_data import DataInput

from Prepare.data_loader import DataLoader
from config.model_parameter import model_parameter
from Model.AttentionTPP import AttentionTPP_MLT

import os
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class Train_main_process:

    def __init__(self):

        start_time = time.time()
        model_parameter_ins = model_parameter()
        data_name = model_parameter_ins.flags.FLAGS.data_name
        self.FLAGS = model_parameter_ins.get_parameter(data_name).FLAGS

        log_ins = create_log(data_name = data_name, model_name = self.FLAGS.model_name, lr = self.FLAGS.learning_rate)

        self.logger = log_ins.logger
        self.logger.info("hello world the experiment begin")
        self.logger.info("the model parameter is : " + str(self.FLAGS.flag_values_dict()))

        prepare_data_ins = DataLoader(self.FLAGS)

        self.logger.info("start loading dataset!")
        self.train_set, self.test_set = prepare_data_ins.load_train_test()

        self.logger.info("dataset loaded!")

        self.logger.info("DataHandle Process cost time: %.2fs" %(time.time() - start_time))
        start_time = time.time()

        self.emb = Type_embedding(is_training=self.FLAGS.is_training,
                                  type_num = self.FLAGS.type_num,
                                  max_seq_len = self.FLAGS.max_seq_len,
                                  sims_len = self.FLAGS.sims_len,
                                  FLAGS = self.FLAGS
                                  )
        self.logger.info('get train test data process cost: %.2fs'%(time.time() - start_time))



    def train(self):
        start_time = time.time()

        if self.FLAGS.per_process_gpu_memory_fraction == 0.0:
            gpu_option = tf.GPUOptions(allow_growth = True)
        elif self.FLAGS.per_process_gpu_memory_fraction == 1.0:
            gpu_option = tf.GPUOptions()
        else:
            gpu_option = tf.GPUOptions(
                per_process_gpu_memory_fraction = self.FLAGS.per_process_gpu_memory_fraction, allow_gwoth = True
            )

        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_DEVISIBLE_DEVICES'] = self.FLAGS.cuda_visible_devices


        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_option))

        if not tf.test.gpu_device_name():
            self.logger.warning('NO GPU is FOUND')
        else:
            self.logger.info(tf.test.gpu_device_name())


        with self.sess.as_default():

            if self.FLAGS.model_name == 'AttentionTPP_MLT':
                self.model = AttentionTPP_MLT(self.FLAGS, self.emb, self.sess)
            self.logger.info('Init finish. cost time: %.2fs' %(time.time() - start_time))



            def eval_model():

                total_event_num = len(self.test_set)
                cross_entropy_lst = []
                for step_i, batch_data in DataInput(self.test_set,self.FLAGS.test_batch_size):
                    step_cross_entropy = self.model.metrics_likelihood(sess = self.sess,
                                                                       batch_data = batch_data)
                    cross_entropy_lst.extend(list(step_cross_entropy))

                avg_cross_entropy_loss = np.sum(cross_entropy_lst) / total_event_num
                return avg_cross_entropy_loss

            self.logger.info('learning rate: %f'%(self.FLAGS.learning_rate))
            self.logger.info('train set: %d'%len(self.train_set))

            self.global_step = 0
            avg_loss = 0.0
            learning_rate = self.FLAGS.learning_rate

            for epoch in range(self.FLAGS.max_epochs):

                random.shuffle(self.train_set)
                for step_i,train_batch_data in DataInput(self.train_set, self.FLAGS.train_batch_size):

                    self.global_step += 1

                    step_loss, merge = self.model.train(self.sess, train_batch_data, learning_rate)
                    self.model.train_writer.add_summary(merge, self.global_step)

                    avg_loss += step_loss


                    if self.global_step % self.FLAGS.display_freq == 0:
                        self.logger.info("epoch: %d, step: %d, global_step: %d" %(epoch, step_i, self.global_step))
                        self.logger.info("train  loss: %.2f"%(avg_loss/ self.FLAGS.display_freq))
                        avg_loss = 0.0

                    if self.global_step % self.FLAGS.eval_freq == 0:
                        test_cross_entropy = eval_model()
                        self.logger.info("cross entropy loss in test set: %.2f"%(test_cross_entropy))

                self.logger.info("epoch : %d"%(epoch))
                test_cross_entropy = eval_model()
                self.logger.info("cross entropy loss in test set: %.2f" % (test_cross_entropy))

            self.save_model()




    def save_model(self):
         is_save_model = False

         if self.global_step % 2000 == 0:
             is_save_model = True

         if is_save_model:
             path = "D://Project/TPP/check_point/" + self.FLAGS.data_name + '-' +\
                    str(self.FLAGS.learning_rate) + "/"
             self.model.save(self.sess,self.global_step, path = path)


if __name__ == "__main__":
    main_process = Train_main_process()
    main_process.train()

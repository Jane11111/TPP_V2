# -*- coding: utf-8 -*-
# @Time    : 2020-08-19 15:03
# @Author  : zxl
# @FileName: train_process.py


import os
import time
import random
import numpy as np
import tensorflow.compat.v1 as tf

from Embedding.history_embedding import history_embedding
from util.model_log import create_log
from DataHandle.get_input_data import DataInput

from Prepare.data_loader import DataLoader
from config.model_parameter import model_parameter
from Model.AttentionTPP import MTAM_TPP_wendy_att_time
from Model.THP import THP
from Model.NHP import NHP
from Model.RMTPP import RMTPP
from Model.SAHP import SAHP
from Model.HP import HP,IHP
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class Train_main_process:

    def __init__(self):

        start_time = time.time()
        model_parameter_ins = model_parameter()
        data_name = model_parameter_ins.flags.FLAGS.data_name
        self.FLAGS = model_parameter_ins.get_parameter(data_name).FLAGS




    def train(self):
        start_time = time.time()

        if self.FLAGS.per_process_gpu_memory_fraction == 0.0:
            gpu_option = tf.GPUOptions(allow_growth = True)
        elif self.FLAGS.per_process_gpu_memory_fraction == 1.0:
            gpu_option = tf.GPUOptions()
        else:
            gpu_option = tf.GPUOptions(
                per_process_gpu_memory_fraction = self.FLAGS.per_process_gpu_memory_fraction, allow_growth = True
            )

        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.FLAGS.cuda_visible_devices


        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_option))



if __name__ == "__main__":
    main_process = Train_main_process()
    main_process.train()

# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 19:29
# @Author  : zxl
# @FileName: data_loader.py

import json
import numpy as np

class DataLoader():

    def __init__(self,FLAGS):
        self.FLAGS = FLAGS
        self.FLAGS = FLAGS
        self.user_count = 1
        self.item_count = 1
        self.category_count = 5
    def pro_time_method(self, time_stamp_seq, mask_time):
        timelast_list = [time_stamp_seq[i+1]-time_stamp_seq[i] for i in range(0,len(time_stamp_seq)-1,1)]
        timelast_list.insert(0,0)
        timenow_list = [mask_time-time_stamp_seq[i] for i in range(0,len(time_stamp_seq),1)]

        return [timelast_list,timenow_list]

    def load_data(self,file_path):
        data_set = []
        count =0
        max_type = 0

        with open(file_path , 'r') as f:
            l = f.readline()
            while l:
                #if count>10000:
                    #break
                tmp = eval(l)
                max_type = max( max_type,np.max(tmp[0]))
                if tmp[4]  < 2:
                    l = f.readline()
                    continue
                tmp.append(self.pro_time_method(tmp[1],tmp[3]))
                sim_time_list = []
                for t in tmp[5]:
                    sim_time_list.append(self.pro_time_method(tmp[1],t))

                tmp.append(sim_time_list)

                data_set.append(tmp)
                l = f.readline()
                count = count+1
        print("max type: %d"%max_type)
        return data_set

    def load_train_test(self):


        origin_train_path = self.FLAGS.in_data_root_path + "train.json"
        origin_test_path =  self.FLAGS.in_data_root_path + "test.json"

        train_path = self.FLAGS.out_data_root_path + "train.txt"
        test_path = self.FLAGS.out_data_root_path + "test.txt"

        if self.FLAGS.split_data:
            self.write_file(origin_train_path, train_path, 'train')
            self.write_file(origin_test_path, test_path, 'test')

        train_set = self.load_data(train_path)
        test_set = self.load_data(test_path)
        print('train len: %d'%len(train_set))
        print('test len: %d'%len(test_set))

        return train_set, test_set

    def process_data(self,seq_lst):
        seq_id = 0
        dataset = []
        max_seq_len = 0
        for event_seq in seq_lst:
            if len(event_seq) == 0:
                continue
            max_seq_len = max(max_seq_len, len(event_seq))
            res = self.process_seq(event_seq, seq_id)
            dataset.extend(res)
            seq_id += 1

        return max_seq_len, dataset

    def write_file(self,in_file, out_file, key):
        with open(in_file, 'r') as f:
            dic = json.load(f)
        seq_lst = dic[key]
        max_seq_len, dataset = self.process_data(seq_lst)
        print(key + ' max_seq_len : %d' % max_seq_len)
        with open(out_file, 'w') as w:
            for seq in dataset:
                w.write(str(seq) + '\n')

    def process_seq(self, event_seq, seq_id):
        """
        把单个事件序列切分成MTAM需要的形式
        :param event_seq:单个事件sequence
        :param seq_id:
        :return:
        """
        res = []
        complete_time_lst = []
        complete_type_lst = []

        for item in event_seq:
            complete_time_lst.append(item['time_since_start'])
            complete_type_lst.append(item['type_event'])

        sims_len = self.FLAGS.sims_len

        # TODO 为什么序列至少需要两个event？？？？？
        for i in range(len(complete_time_lst)):
            target_type = complete_type_lst[i]
            target_time = complete_time_lst[i]

            if i == 0: #TODO  使用历史生成h，接着把target与h通过全连接网络
                seq_len = 1
                type_lst = [0]
                time_lst = [0]
            # elif i == 1:
            #     seq_len = 2
            #     type_lst = [ 0,complete_type_lst[0]]
            #     time_lst = [ 0,complete_time_lst[0]]

            else:
                seq_len = min(i,self.FLAGS.max_seq_len)
                start_idx = max(0, i-seq_len)
                end_idx = i
                type_lst = complete_type_lst[start_idx:end_idx]
                time_lst = complete_time_lst[start_idx:end_idx]
            sims_time_lst = list(np.random.uniform(time_lst[-1],target_time, sims_len))
            res.append([type_lst, time_lst,
                        target_type, target_time, seq_len, sims_time_lst])
            # res.append(sample_time_list)
        return res




# if __name__ == "__main__":
#
#     loder = DataLoader()
#     train_set, test_set = load_train_test()
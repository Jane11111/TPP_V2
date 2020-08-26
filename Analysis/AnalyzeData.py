# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 10:46
# @Author  : zxl
# @FileName: AnalyzeData.py

import  json
import matplotlib.pyplot as plt

"""
统计训练集、测试集中的数据信息
最长、最短、平均 seq 长度
所有type类型，分布
样本总数
"""

if __name__ == "__main__":

    root = "D://Project/TPP_V2/data/training_testing_data/data_mimic/total/"
    sample_count = 0
    train_test_count = [0,0]
    total_seq_len = 0
    max_seq_len = 0
    min_seq_len = 1000
    type_dic = {}

    for idx, filename in enumerate(['train.txt','test.txt']):
        file_path = root + filename
        key = filename[:-4]


        with open(file_path,'r') as f:
            l = f.readline()
            while l:
                arr = eval(l)
                target_type=arr[2]
                seq_len = arr[4]
                train_test_count[idx] += 1
                sample_count+=1
                total_seq_len+=seq_len
                max_seq_len = max(max_seq_len,seq_len)
                min_seq_len = min(min_seq_len,seq_len)
                if target_type not in type_dic:
                    type_dic[target_type] = 0
                type_dic[target_type] +=1
                l = f.readline()


        sorted_dic = sorted(type_dic.items(), key=lambda x: x[1], reverse=True)
        type_name = []
        type_frequency = []
        for item in sorted_dic:
            type_name.append(item[0])
            type_frequency.append(item[1])

        plt.bar(range(len(type_frequency)),type_frequency)
        plt.xticks(range(len(type_frequency)),type_name)
        plt.title(key)
        plt.xlabel('type_num')
        plt.ylabel('type_frequency')
        plt.show()
    print(' count: %d, train count : %d, test count: %d,type_num:%d, max seq len: %d, min seq len: %d,avg seq len: %.2f' % \
          (sample_count, train_test_count[0],train_test_count[1],len(type_dic), max_seq_len, min_seq_len, total_seq_len / sample_count))


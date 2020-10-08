# -*- coding: utf-8 -*-
# @Time    : 2020/8/24 10:33
# @Author  : zxl
# @FileName: merge_data.py

"""
将5个mimic数据集合并起来
"""

import json
import pickle

if __name__ == "__main__":

    root = "D://Data/Hawkes/data_mimic/"
    out_root = root + 'total/'

    for filename in ['train.pkl', 'test.pkl','dev.pkl']:
        out_path = out_root+filename
        new_dic = {"dim_process": 75,
                   "train":[],
                   "dev":[],
                   "test":[],
                   "args":None,
                   "devtest":[]}
        key = filename[:-4]
        for dirname in ['fold1','fold2','fold3','fold4','fold5']:
            in_path = root + dirname +'/' + filename
            with open(in_path,'rb') as f:
                json_obj = pickle.load(f, encoding='latin-1')
            new_dic[key].extend(json_obj[key])

        pickle.dump(new_dic, open(out_path, 'wb' ))




# -*- coding: utf-8 -*-
# @Time    : 2020-08-23 22:58
# @Author  : zxl
# @FileName: pkl2json.py


import json
import time
import numpy as np
import pickle


def convert_lst(lst):
    new_lst = []
    for seq in lst:
        new_seq = []
        for event in seq:
            new_event = {}
            for k in event:
                if type(event[k]) == np.int32:
                    new_event[k] = int(event[k])
                else:
                    new_event[k] = float(event[k])
            new_seq.append(new_event)
        new_lst.append(new_seq)
    return new_lst




if __name__ == "__main__":

    root = "D://Project/TPP_V2/data/origin_data/data_mimic/"
    for folder_name in ['fold1','fold2','fold3','fold4','fold5']:
        path = root + folder_name

        for file_name in ['train.pkl','dev.pkl','test.pkl']:
            name = file_name[:-4]
            in_path = path + '/' + file_name
            out_path = path + '/'+file_name[:-4]+'.json'
            with open(in_path,'rb') as f:
                dic = pickle.load(f)
            dic[name] = convert_lst(dic[name])
            dic['dim_process'] = int(dic['dim_process'])
            json_str = json.dumps(dic,indent=4)
            with open(out_path,'w') as w:
                w.write(json_str)



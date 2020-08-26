# -*- coding: utf-8 -*-
# @Time    : 2020/8/24 10:33
# @Author  : zxl
# @FileName: merge_data.py

"""
将5个mimic数据集合并起来
"""

import json

if __name__ == "__main__":

    root = "D://Project/TPP_V2/data/origin_data/data_mimic/"
    out_root = root + 'total/'

    for filename in ['train.json', 'test.json']:
        out_path = out_root+filename
        new_dic = {"dim_process": 75,
                   "train":[],
                   "dev":[],
                   "test":[],
                   "args":None,
                   "devtest":[]}
        key = filename[:-5]
        for dirname in ['fold1','fold2','fold3','fold4','fold5']:
            in_path = root + dirname +'/' + filename
            with open(in_path,'r') as f:
                json_obj = json.load(f)
            new_dic[key].extend(json_obj[key])

        json_str = json.dumps(new_dic,indent=4)
        with open(out_path,'w') as w:
            w.write(json_str)




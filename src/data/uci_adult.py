# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:31:58 2021

@author: iwenc
"""

import os
import pandas as pd
if __name__ == "__main__":
    import util # 单独运行本脚本用
else:
    from . import util  # 当作一个module被别人 import 时用 


def load(data_dir="../../dataset", drop_useless=True, sampling_cfg=None):
    # 1. read/uncompress data
    file_name = os.path.join(data_dir,'UCI/Adult/adult.data')
    print (f'{file_name=}')
    print(pd.show_versions())
    print('=======================')
    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
               'marital-status', 'occupation', 'relationship', 'race', 'sex', 
               'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country','predclass']
    training_set = pd.read_csv(file_name, header=None, names=headers, sep=',\s',
                               na_values=["?"],on_bad_lines='warn', engine='python')
        
    file_name = os.path.join(data_dir,'UCI/Adult/adult.test')
    test_set = pd.read_csv(file_name, header=None, names=headers,  sep=',\s',
            na_values=["?"],on_bad_lines='error', engine='python', skiprows=1)

    df = pd.concat([training_set,test_set])

    
    # 2. convert numeric/categorical columns
    num_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']
    for col in num_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].to_numeric()
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype('category')
    
    # 3. simple feature extraction

    
    # 4. compute class label
    y_col = 'predclass'
    f = lambda y: 1 if y=='>50K' or y=='>50K.'  else 0
    df[y_col] = df[y_col].map(f) 

    # 5. drop_useless
    if drop_useless:
        # a. manually remove useless columns
        
        # b. auto identified useless columns
        useless_cols_dict = util.find_useless_colum(df)
        df = util.drop_useless(df, useless_cols_dict)

    # 6. sampling by class
    if sampling_cfg:
        df = util.sampling_by_class(df, y_col, sampling_cfg)

        # remove categorical cols with too few samples_per_cat
            
    return df, y_col

if __name__ == "__main__":
    df,y_col = load()
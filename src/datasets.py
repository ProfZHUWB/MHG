# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:00:29 2021

@author: iwenc
"""

data_dir = '../dataset'
save_dir = '../results/datasets'

import os
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
import warnings


import preprocessor


# import colorama
# replaced by warnings.warn()
# def print_warning(msg):
#     print(f'{colorama.Fore.YELLOW}{msg}{colorama.Style.RESET_ALL}')

# def to_categorical(df, col):
#     if not pd.api.types.is_categorical_dtype(df[col]):
#         df[col] = df[col].astype('category')

# def to_numeric(df, col):
#     if not pd.api.types.is_numeric_dtype(df[col]):
#         df[col] = df[col].to_numeric()



class BCDataSet:
    def __init__(self, name, df, y_col):
        self.name = name
        self.y = df[y_col]
        vc = self.y.value_counts(sort=False,dropna=False)

        if vc.size > 2:
            raise RuntimeError(f"target variable {y_col} has more than 2 values: {vc}")
        if not pd.api.types.is_integer_dtype(self.y):
            raise RuntimeError(f"target variable {y_col} is not integers")
        
        self.X = df.drop(y_col, axis=1)
            
    def get_largest_card_cat_var(self):
        hc_col = None
        hc_card = -1
        cats = self.X.select_dtypes(include='category')
        for col in cats:
            value_counts = self.X[col].value_counts(sort=False,dropna=False)
            if value_counts.size > hc_card:
                hc_col = col
                hc_card = value_counts.size

        return hc_col

    def rf_importance_df(self,seed=3,save_dir=save_dir,load_existing=True):       
        prefix = os.path.join(save_dir,self.name+"_imp"+"_s="+str(seed))
        filename = prefix+".csv"
        if load_existing and os.path.exists(filename):
            return pd.read_csv(filename, index_col=0)
        
        X = self.X.copy(deep=True)
        
        # 1. numerical variable
        #        replace inf by max+1
        #        replace -inf by min-1
        #        filling na by mean (excluding na)
        # 2. encode categorical variable by TargetEncoding
        for col in X:
            if pd.api.types.is_numeric_dtype(X[col]):
                col_max = X[col].max()
                if col_max == np.inf:
                    msg = f'Data set: {self.name} col: {col} np.inf -> max+1'
                    warnings.warn(msg)
                    new_col = X[col].replace([np.inf], np.nan)
                    col_max = new_col.max()
                    X[col].replace([np.inf], col_max+1, inplace=True)
                col_min = X[col].min()
                if col_min == -np.inf:
                    msg = f'Data set: {self.name} col: {col} -np.inf -> min-1'
                    warnings.warn(msg)
                    new_col = X[col].replace([-np.inf], np.nan)
                    col_min = new_col.min()
                    X[col].replace([-np.inf], col_min-1, inplace=True)
                
                v = X[col].mean()
                X[col] = X[col].fillna(v)
            elif pd.api.types.is_categorical_dtype(X[col]):
                X[col] = TargetEncoder(cols=col).fit_transform(X[col],self.y)
                assert len(X[col].shape) == 1  # 1D array
                # print(f'{X[col].shape=}')
        
        clf = RandomForestClassifier(random_state=seed,
                                     n_estimators = 100,     # default 100
                                     criterion = 'gini',     # default 'gini'
                                     max_depth = 30,         # default None, no limit
                                     min_samples_split = 20, # default 2
                                     min_samples_leaf = 1)   # default 1
        clf.fit(X, self.y)
        df_imp = pd.DataFrame()
        df_imp['mdi_imp'] = pd.Series(clf.feature_importances_, index=self.X.columns)
        
        result = permutation_importance(
            clf, X, self.y, n_repeats=10, random_state=42, n_jobs=1
        )
        df_imp['perm_imp'] = pd.Series(result.importances_mean, index=self.X.columns)
        os.makedirs(save_dir,exist_ok=True)
        df_imp.to_csv(filename)
        
        # plot figure
        importance = df_imp['mdi_imp'].sort_values(ascending=True)
        plt.style.use('seaborn-whitegrid')
        importance.plot(kind='barh', figsize=(20,len(importance)/2))
        plt.xlabel('var importance MDI')
        plt.savefig(prefix+"-mdi.png")
        plt.show()

        importance = df_imp['perm_imp'].sort_values(ascending=True)
        plt.style.use('seaborn-whitegrid')
        importance.plot(kind='barh', figsize=(20,len(importance)/2))
        plt.xlabel('var importance permutation')
        plt.savefig(prefix+"-perm.png")
        plt.show()
        
        return df_imp

    def stats_df(self,seed=3,save_dir=save_dir,load_existing=True,compute_imp=True):

        var2card = {}
        var2avg_samples_per_cat = {}   # cat_variable to samples_per_cat
        cats = self.X.select_dtypes(include='category')
        for col in cats:
            value_counts = self.X[col].value_counts(sort=False,dropna=False)
            var2card[col] = value_counts.size
            var2avg_samples_per_cat[col] = self.X[col].size / value_counts.size

        self.stats = pd.DataFrame(data={'cardinality':pd.Series(data=var2card, index=self.X.columns)})
        card = self.stats['cardinality']
        self.stats['card_rank'] = card[card.isna()==False].rank(ascending=False)
        self.stats['samples_per_cat'] = pd.Series(data=var2avg_samples_per_cat, index=self.X.columns)
        
        self.stats['na_ratio'] = self.X.isna().sum()/max(self.X.shape[0],1) # correctly handle empty table
        
        self.stats['positve_ratio'] = self.y.sum()/self.y.size

        if compute_imp:        
            df_imp = self.rf_importance_df(seed=seed,save_dir=save_dir,load_existing=load_existing)
            self.stats['mdi_imp'] = df_imp['mdi_imp']
            self.stats['perm_imp'] = df_imp['perm_imp']
            imp = self.stats['mdi_imp']
            self.stats['mdi_imp_rank'] = imp.rank(ascending=False)
            self.stats['mdi_imp_rank_cat'] = imp[self.stats['cardinality'].isna()==False].rank(ascending=False)

            imp = self.stats['perm_imp']
            self.stats['perm_imp_rank'] = imp.rank(ascending=False)
            self.stats['perm_imp_rank_cat'] = imp[self.stats['cardinality'].isna()==False].rank(ascending=False)
        
        if save_dir is not None:
            os.makedirs(save_dir,exist_ok=True)
            filename = os.path.join(save_dir,self.name+"_s="+str(seed)+".csv")
            self.stats.to_csv(filename)
        return self.stats
    


# Define a dictionary of data sets
# Key is the name of a data set
# Value is a function that load data set
from data import uci_adult,tianchi_auto_loan_default_risk,misc_colleges,lacity_crime
from data import employee_salaries,h1b_visa,traffic_violation,road_safety_accident

data_loader_dict = {
    'Adult':uci_adult.load,
    'Colleges':misc_colleges.load,
    'AutoLoan':tianchi_auto_loan_default_risk.load,
    'Crime':lacity_crime.load,
    'EmployeeSalaries':employee_salaries.load,
    'H1BVisa':h1b_visa.load,
    # 'LendingClub': lending_club.load,
    'TrafficViolations': traffic_violation.load,
    'RoadSafety': road_safety_accident.load
# TODO: add more sets
}




def load_one_set(name, drop_useless=True, no_sampling=False, fillna=False, load_existing=True, seed=3):
    loader = data_loader_dict[name]
    if no_sampling:
        df,y_col = loader(data_dir=data_dir, drop_useless=drop_useless,sampling_cfg=None)
    else:
        df,y_col = loader(data_dir=data_dir, drop_useless=drop_useless)
    base_name = name
    if not drop_useless:
        name += '_full'
    if no_sampling:
        name += '_nosampling'
    if fillna:
        name += '_fillna'
    ds = BCDataSet(name, df, y_col)
    ds.base_name = base_name
    # compute var importance only on small samples
    ds.stats_df(seed=seed, load_existing=load_existing, compute_imp=(not no_sampling))
    print("----------------")
    print(f'hc_cat_name: {ds.get_largest_card_cat_var()}')
    if fillna:
        ds.X = preprocessor.fillna(ds.X)
    return ds

def load_all(drop_useless=True, no_sampling=False, fillna=False, load_existing=True, seed=3):
    for name,loader in data_loader_dict.items():
        yield load_one_set(name, drop_useless=drop_useless, no_sampling=no_sampling, fillna=fillna,load_existing=load_existing, seed=seed)




if __name__ == "__main__":
    
    # 检查 df 的 categorical 列是否全是 int 或者 float 型。
    # 1. 数据集中分类型变量通常会用字符串表示每个水平的名字，但是有些比赛会把每个水平替
    #    换成一个整数。
    # 2. 由于现在的代码先用 SimpleImputer 对数据集 df 中的所有分类变量进行填充，
    #    SimpleImputer 会把 DataFrame 变成 numpy array, 如果所有列都是 int 或者
    #    float 类型 numpy array 的类型也是 int 或者 float 类型
    # 3. 当 TargetEncoder 输入为 numpy array 时，只有类型为 object 才会当作分类变量
    #    进行编码，其它类型包括 int 类型的，会不进行编码。
    # 4. 当 df 中所有分类型变量都是 int 类型，就会不进行 Target 编码。
    # 解决方法：
    # 1. 目前只有 AutoLoan 数据集的所有 categorical 列是 int 型，
    #    在 data/tianchi_auto_default_risk.py中把 int 型的分类变量变成 str 型
    # 2. TODO: 替换 SimpleImputer 
    #    更新 preprocessor.py： 删掉 SimpleImputer，在进入 pipeline 之前，
    #    根据列的属性填充缺失值。为每个 categorical 列用 value_count 找到最大频次
    #    的值然后用 df[col].fillna(value, inplace= True) 手工填充
    def cat_col_int_float_check(df):    
        cat_col_counts = 0
        cat_col_int_counts = 0
        for col in df:
            if pd.api.types.is_categorical_dtype(df[col]):
                cat_col_counts += 1
                a = df[col].to_numpy()
                if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, float):
                    # df[col] = df[col].astype('str').astype('category')
                    cat_col_int_counts += 1 
                                        
        if cat_col_counts == cat_col_int_counts:
            raise RuntimeError('All categorical variable type is integer/float, need to convert it to str')
    
    
    tbl_stats = pd.DataFrame(columns=['hc_col'])
    
    for ds in load_all():
    # for ds in load_all(load_existing=False): # force re-compute statistics 
        hc_col = ds.get_largest_card_cat_var()
        tbl_stats.loc[ds.base_name,'hc_col'] = hc_col
        for col in ds.stats:
            tbl_stats.loc[ds.base_name,col] = ds.stats.loc[hc_col,col]
        cat_col_int_float_check(ds.X)
        tbl_stats.loc[ds.base_name,'positive samples'] = ds.y.sum()
        tbl_stats.loc[ds.base_name,'total samples'] = len(ds.y)
        print(f'------- loaded data set {ds.name} -----------')

    # compute the variable importance after fillna
    for ds in load_all(fillna=True):
        hc_col = ds.get_largest_card_cat_var()
        tbl_stats.loc[ds.base_name,'importance-fillna'] = ds.stats.loc[hc_col,'importance']

    for ds in load_all(no_sampling=True, fillna=True):
        tbl_stats.loc[ds.base_name,'p_count'] = ds.y.sum()
        tbl_stats.loc[ds.base_name,'N'] = ds.X.shape[0]
        tbl_stats.loc[ds.base_name,'X_cols'] = ds.X.shape[1]
        
    tbl_stats.to_csv(os.path.join(save_dir,"summary.csv"))


# # TODO: test added set

#     # ds = load_one_set('H1BVisa')
#     # stats = ds.stats_df()
#     # print(f'hc_cat_name: {ds.get_largest_card_cat_var()}')
#     # ds.X = preprocessor.fillna(ds.X)

    

    

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 20:41:14 2019

@author: Xiaoting Wu, Wenbin Zhu and Ying Fu

"""


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import pickle

import encoder
import warnings


def savePkl(obj, fileName):
    dirName = os.path.dirname(fileName)
    if len(dirName) > 0:
        os.makedirs(dirName, exist_ok=True)
    with open(fileName, "wb") as file:
        pickle.dump(obj,file)

def loadPkl(fileName):
    with open(fileName, "rb") as file:
        return pickle.load(file)


def fillna(df, value_for_entire_missing_col=0):
    '''
    Filling missing values column by column.
    1. For numeric columns, fill missing values by median, this is more robust than
        average when there are extreme values;
    2. For other columns, fill missing values by the most frequent values.
        For categorical variable, missing value are replaced by the largest category.

    Parameters
    ----------
    df : pandas.DataFrame
        A table that main contains missing values.
    value_for_entire_missing_col : object, optional
        If all values in a column are missing, we use this value. The default is 0.

    Returns
    -------
    df_copy : pandas.DataFrame
        A copy of df with missing value filled.

    '''
    df_copy = df.copy()
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median(skipna=True)
            if np.isnan(median):
                median = value_for_entire_missing_col
            df_copy[col] = df[col].fillna(median)
        else: # pd.api.types.is_categorical_dtype(df[col]), and other types
            vc = df[col].value_counts(sort=True,dropna=True)
            if vc.size > 0:
                df_copy[col] = df[col].fillna(vc.index[0])
            else:
                df_copy.loc[col] = value_for_entire_missing_col
    return df_copy

    # Use column transfomer the resulting output is numpy.ndarray
    # numeric_transformer = Pipeline(steps=[
    #         ('imputer', SimpleImputer(strategy='median'))])
    # categorical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='most_frequent'))])

    # return ColumnTransformer(transformers=[
    #         ('num', numeric_transformer, num_cols),
    #         ('cat', categorical_transformer, cat_cols)],
    #         remainder = 'drop') 

def one_hot():
    return OneHotEncoder(handle_unknown='ignore', # unknown value in transform is encoded as zero vector
                         # drop='first',          # remove the first category to avoid collinearity
                         sparse = False)          # return a dense matrix

# debug. Insert code before calling fit() and transform()
# class MyIm(SimpleImputer):
    # def transform(self, X, **kwargs):
    #     print("--------- simple transform ----------")
        
    #     X_train = SimpleImputer.transform(self, X, **kwargs)
    #     print(f'{X.shape=}')
    #     print(f'{X_train.shape=}')
    #     print(f'{type(X)=}')
    #     print(f'{type(X_train)=}')
    #     print(f'{X.iloc[479,:]=}')
    #     print(f'{X_train[479]=}')
        
    #     return X_train
        
# debug. Insert code before calling fit() and transform()
# class MyTE(TargetEncoder):
    # def fit(self, X,y,**kwargs):
    #     print("xxxx         hhhhhh ---")
    #     print(f'{type(y)=}')
    #     print(f'{y.iloc[479]=}')
        
    #     from category_encoders import OrdinalEncoder
    #     self.ordinal_encoder = OrdinalEncoder(
    #         verbose=self.verbose,
    #         cols=self.cols,
    #         handle_unknown='value',
    #         handle_missing='value'
    #     )
    #     self.ordinal_encoder = self.ordinal_encoder.fit(X)
    #     X_ordinal = self.ordinal_encoder.transform(X)
    #     self.mapping = self.fit_target_encoding(X_ordinal, y)
    #     print(f'{self.mapping[0]=}')
    #     print(f'{self.ordinal_encoder.category_mapping[0]["mapping"]=}')
    #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    #     print(f'{y.value_counts()=}')
    #     # X_ordinal['Y'] = y.values
    #     X_ordinal['Y'] = y
    #     X_ordinal.to_csv('test_te.csv', index=False)
    #     raise RuntimeError

    #     col = 0
    #     stats = y.groupby(X_ordinal[col]).agg(['count', 'mean'])   
    #     print(f'{stats=}')
        
    #     raise RuntimeError
        
    #     abc = TargetEncoder.fit(self, X, y, **kwargs)
    #     print(f'{self.mapping[0]=}')
    #     print(f'{self.handle_missing=}')
    #     print(f'{self.handle_unknown=}')
    #     print("$$$$$$$$$$$$$$$$$$$$$")
    #     print(f'{self.ordinal_encoder.category_mapping[0]["mapping"]=}')
    #     return abc
    
    # def transform(self, X, y=None, **kwargs):
    #     X_train = TargetEncoder.transform(self, X, y, **kwargs)
        
    #     print(f'{X.shape=}')
    #     print(f'{X_train.shape=}')
    #     print(f'{type(X)=}')
    #     print(f'{type(X_train)=}')
        
    #     X_int = self.ordinal_encoder.transform(X)
    #     print(f'{type(X_int)=}   !!!!!!!!!!!!!!!')
    #     print(f'{X.loc[479]=}')
    #     print(f'{X_int.loc[479]=}')
        
    #     X_2 = self.target_encode(X_int)
    #     print(f'{type(X_int)=}   &&&&&&&&&&&&&&&&&&')
    #     print(f'{X_2.loc[479]=}')
        
    #     for row in range(X_train.shape[0]):
    #         if np.any(np.isnan(X_train.loc[row,:])):
    #             col_idx = np.where(np.isnan(X_train.loc[row,:]))
    #             print(f'{row=}')
    #             print(f'{col_idx=}')
    #             print(f'{X.loc[row]=}')
    #             print(f'{X_train.loc[row,:]=}')
    #             print(f'{self.mapping[0]=}')
    #             print(f'{self.ordinal_encoder.category_mapping[0]["mapping"]=}')
                
    #             raise RuntimeError
    #     return X_train
        

def prepare_train_test(ds, train_index, test_index, 
                       hc_col, hc_cat_encoder='pca', hc_group = 10, other_cat_encoder='one-hot',
                       fold = 0,
                       save_dir='../results/datasets', load_existing = True,
                       allow_nan_inf = False,
                       verbose=False):
    '''
    Columns of ds.X is classified into four types:
        numeric:   Numerical columns
        hc_col:    Categorical column with high cardinality
        other_cat: Categorical columns other than hc_col
        other:     Other columns, they are dropped

    Each numeric column is preprocessed as follows:
        0. replace np.inf by max_excluding_inf+1, -np.inf by min_excluding_-inf -1
        1. fill missing value by median in training set
        2. covert to z-score via formula (x-mean) / std
    Each other_cat column is preprocessed as follows:
        1. fill missing value by largest category in training set
        2. if other_cat_encoder == 'one-hot' (the approperiate scheme for most learning models)
                apply one-hot encoding
           otherwise (the approperiate scheme for tree based learning models such as DecisionTree, RandomForest, GBDT)
                apply TargetEncoding (https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
                using Micci-Barreca's formula: https://dl.acm.org/doi/10.1145/507533.507538
                    X_i ==> S_i, where S_i is an estimate of P(Y=1|X=X_i)      (1)
                    S_i = \lambda(n_i) P(Y=1|X=X_i) + (1-\lambda(n_i)) P(Y=1)  (3)
                          P(Y=1|X=X_i) = n_{iY} / n_i, the proportion of samples with Y=1 in the category X=X_i
                          P(Y=1) = n_Y / n_TR, the proprotin of samples with Y=1 in the entire training set
                          \lambda(n) = 1 / (1 + exp(-(n-k)/f) ), is S-shaped curve with value 0.5 when n = k and approaching 1 when n > k and approaching 0 when n -> 0
                S_i is the blending of posterior and prior probability and when
                the number of samples in the category n_i become larger, the posterior is more reliable and use more of it
                
                In the implementation of TargetEncoder,
                    n_i = 1  ==> S_i = prior = y.mean()                        
                    n_i > 1  ==> S_i = smoov * mean_y_in_category + (1-smoov) * prior
                                       smoov = 1 / (1 + exp(- (#sample_in_cat - min_samples_leaf=1) / smoothing=1.0))
    The hc_col column is preprocessed as follows:
        1. fill missing value by largest category in training set
        2. if hc_cat_encoder == 'pca':
                apply one-hot encoder, then apply pca to reduce dimension to hc_group
           otherwise
                apply MaxHomoEncoder to reduce groups to hc_group, then apply one-hot encoder
                    normal category are assigned to hc_group groups
                    category with samples <= 2 are merged into CAT_TYPICAL
                    missing category are merged into CAT_TYPICAL
                    unseen category in testing data are treated as CAT_TYPICAL

    Parameters
    ----------
    ds : datasets.BCDataSet
        A data set
    train_index : row indices
        ds.X[train_index] gives the X of training set
        ds.y[train_index] gives the y of training set
    test_index : row indices
        ds.X[test_index] gives the X of test set
        ds.y[test_index] gives the y of test set
    hc_col : string
        the name of high cardinality categorical column
    hc_cat_encoder : string, optional
        The encoding scheme for high cardinality categorical column, either 'pca' or 'mhe'.
        'pca': apply one-hot then pca to reduce to hc_group dimension;
        'mhe': apply MaxHomoEncoder to reduce to hc_group categories then apply one-hot.
        'mhe+te': apply MaxHomoEncoder to reduce to hc_group categories then apply TargetEncoder
        'one-hot': ignore hc_group, encode all categories by one-hot
        'TargetEncoder': ignore hc_group, encode all categories by TargetEncoder
        The default is 'pca'.
    hc_group : int, optional
        The number of groups to encode the high cardinality categorical variable.
        It must not exceed the cardinality. The default is 10.
    other_cat_encoder : string, optional
        The encoding scheme for other_cat columns, either 'one-hot' or 'TargetEncoder'.
        The default is 'one-hot'.
    save_dir : string, optional
        Directory for saving processed data sets. The default is '../result/datasets'.
    load_existing : bool, optional
        If True and save_dir/{ds.basename}.csv exists, load the existing file.
        Otherwise, preprocess the dataset and save to save_dir/{ds.basename}.pkl
        The default is True.
    verbose : bool, optional
        If True print more debug information. The default is False.

    Returns
    -------
    X_train : numpy.ndarray 2D
        encoded training data X
    y_train : numpy.ndarray 1D
        training label y
    X_test : numpy.ndarray 2D
        encoded test data X
    y_test : numpy.ndarray 1D
        training label y
    '''
    
    filename = os.path.join(save_dir,f'{ds.name}_hc={hc_col}_enc={hc_cat_encoder}_J={hc_group}_oenc={other_cat_encoder}_fold={fold}.pkl')
    if load_existing and os.path.exists(filename):
        if verbose:
            print(f'load from {filename}')
        return loadPkl(filename)

    if ds.y.isna().sum() != 0:
        raise ValueError(f'{ds.y.isna().sum()=} is not zero, there are missing y values, please delete such records or fill in missing values')
    
    vc = ds.X[hc_col].value_counts() 
    cardinality = (~vc.index.isna()).sum()
    if cardinality < hc_group:
        hc_group = cardinality
        # when ratio is close to 1.0, hc_group ~= cardinality in original data set
        # after sampling cardinality in train data become less than hc_group, in this case
        # we refine hc_group
        # raise ValueError(f'{hc_group=} exceeds {cardinality=} of {hc_col=}')

    num_cols = [col for col in ds.X if pd.api.types.is_numeric_dtype(ds.X[col])]
    cat_cols = [col for col in ds.X if pd.api.types.is_categorical_dtype(ds.X[col]) and col != hc_col]  
    if verbose:
        print(f'{len(num_cols)=}, {num_cols=}')
        print(f'{len(cat_cols)=}, {cat_cols=}')
    
    for col in num_cols:
        col_max = ds.X[col].max()    
        if col_max == np.inf:
            warnings.warn(f'Data set: {ds.name} col: {col} np.inf -> max+1')
            new_col = ds.X[col].replace([np.inf], np.nan)
            col_max = new_col.max()
            ds.X[col].replace([np.inf], col_max+1, inplace=True)

        col_min = ds.X[col].min()
        if col_min == -np.inf:
            warnings.warn(f'Data set: {ds.name} col: {col} -np.inf -> min-1')
            new_col = ds.X[col].replace([-np.inf], np.nan)
            col_min = new_col.min()
            ds.X[col].replace([-np.inf], col_min-1, inplace=True)

    
    # Must reset_index, for pipline to work correctly
    # 1. X = ds.X.iloc[train_index,:], the index for rows are train_index
    #    integer but not continuous, and may not starts from 1
    # 2. SimpleImputer.transform(X) will convert X into np.ndarray
    # 3. In pipeline, TargetEncoder.fit(X), .transform(X) will convert X
    #    (np.ndarray) into pd.DataFrame, which will create new index 1,2,3...
    # 4. TargetEncoder.fit(X, y) will invoke y.groupby(X[col]) to 
    #    group y by X[col]. In this step, index is used to match values
    #    in y with values in X[col]. Because the index of X has been
    #    reset to 1,2,3,..., for some index, there is no corresponding y value
    df_train = ds.X.iloc[train_index,:].reset_index(drop=True)
    s_train = ds.y.iloc[train_index].reset_index(drop=True)
    df_test = ds.X.iloc[test_index,:].reset_index(drop=True)
    s_test = ds.y.iloc[test_index].reset_index(drop=True)
    if verbose:
        print(f'{df_train.shape=}')
        print(f'{s_train.shape=}')
        print(f'{df_test.shape=}')
        print(f'{s_test.shape=}')

    # transform numeric columns
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # transform categoricals (excluding hc_col)
    if other_cat_encoder == 'one-hot':
        cat_encoder = one_hot()
    else: # 'TargetEncoder'
        # print("xxx---------------xxxxx-----------")
        # cat_encoder = MyTE()
        cat_encoder = TargetEncoder() # when X is ndarray, encode all columns; when X is DataFrame, encode dtype='categor' and 'object'
    
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', cat_encoder)])

    # transform hc_col
    if hc_cat_encoder == 'pca': # one-hot + PCA
        hc_pip = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one-hot', one_hot()),
            ('pca', PCA(n_components=hc_group))])
    elif hc_cat_encoder == 'mhe': # MHG + one-hot
        col_conf = encoder.ColumnConfig(group_count = hc_group, idx = 0,
                                min_samples_per_cat=3,
                 insufficient_handler = encoder.CategoryHandler.CAT_TYPICAL,
                 missing_handler = encoder.CategoryHandler.CAT_TYPICAL, # we already filled missing values, no need for the special group of missing
                 unseen_handler = encoder.CategoryHandler.CAT_TYPICAL)
        mhe_encoder = encoder.MaxHomoEncoder([col_conf])
        hc_pip = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('mhe', mhe_encoder),
            ('one-hot', one_hot())])
    elif hc_cat_encoder == 'mhe+te': # MHG + TargetEncoding, suitable for tree based methods
        col_conf = encoder.ColumnConfig(group_count = hc_group, idx = 0,
                                min_samples_per_cat=3,
                 insufficient_handler = encoder.CategoryHandler.CAT_TYPICAL,
                 missing_handler = encoder.CategoryHandler.CAT_TYPICAL, # we already filled missing values, no need for the special group of missing
                 unseen_handler = encoder.CategoryHandler.CAT_TYPICAL)
        mhe_encoder = encoder.MaxHomoEncoder([col_conf])
        hc_pip = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('mhe', mhe_encoder),
            ('target-encoding', TargetEncoder())])
    elif hc_cat_encoder == 'one-hot': # one-hot, no grouping or dimension reduction
        hc_pip = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one-hot', one_hot())])
    elif hc_cat_encoder == 'TargetEncoder':
        hc_pip = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one-hot', TargetEncoder())])
    else:
        raise RuntimeError(f'unsupported hc_cat_encoder: {hc_cat_encoder}')
    
    trans = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('hc', hc_pip, [hc_col])], remainder = 'drop') 
    trans.fit(df_train, s_train)
    X_train = trans.transform(df_train)
    X_test = trans.transform(df_test)
    if verbose:
        print(f'{X_train.shape=}')
        print(f'{X_train.shape=}')
        print(f'save to {filename}')


    # debug: find        
    # for row in range(X_train.shape[0]):
    #     if np.any(np.isnan(X_train[row])):
    #         print(f'{row=}')
    #         print(f'col: {np.where(np.isnan(X_train[row]))[0]=}')
    #         print(f'{X_train[row,18]=}')
    #         print(f'{df_train.loc[train_index[row],cat_cols[0]]=}')
    #         # print(f'{df_train.info()}')
    #         print(f'{df_train.iloc[row,:]=}')
    #         print(f'{hc_cat_encoder=}')
    #         print(f'{hc_col=}')
    #         print(f'{other_cat_encoder=}')
    #         print(f'{cat_cols[0]=}')
    #         raise RuntimeError

    y_train = s_train.to_numpy()
    y_test = s_test.to_numpy()
    
    if not allow_nan_inf:
        if np.sum(np.isnan(X_train)) > 0:
            raise ValueError('X_train contains NaN')
        if np.sum(np.isinf(X_train)) > 0:
            raise ValueError('X_train contains np.inf or -np.inf')
        if np.sum(np.isnan(y_train)) > 0:
            raise ValueError('y_train contains NaN')
        if np.sum(np.isinf(y_train)) > 0:
            raise ValueError('y_train contains np.inf or -np.inf')
        if np.sum(np.isnan(X_test)) > 0:
            raise ValueError('X_test contains NaN')
        if np.sum(np.isinf(X_test)) > 0:
            raise ValueError('X_test contains np.inf or -np.inf')
        if np.sum(np.isnan(y_test)) > 0:
            raise ValueError('y_test contains NaN')
        if np.sum(np.isinf(y_test)) > 0:
            raise ValueError('y_test contains np.inf or -np.inf')

    result = X_train, y_train, X_test, y_test
    savePkl(result, filename)
    return result


if __name__ == '__main__':

    import datasets    
    ds = datasets.load_one_set('Adult')
    hc_col = ds.get_largest_card_cat_var()
    
    from sklearn.model_selection import ShuffleSplit
    rs = ShuffleSplit(n_splits=1, test_size=.30, random_state=0)
    for train_index, test_index in rs.split(ds.X):
        print('------ one-hot   pca --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('------ one-hot   mhe --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_cat_encoder='mhe', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('------ one-hot   mhe+te --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_cat_encoder='mhe+te', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('------ one-hot   one-hot --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_cat_encoder='one-hot', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('------ one-hot   TargetEncoder --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_cat_encoder='TargetEncoder', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('------ TargetEncoder   pca --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, other_cat_encoder='TargetEncoder', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('------ TargetEncoder   mhe --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_cat_encoder='mhe', other_cat_encoder='TargetEncoder', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0
        
        print('------ TargetEncoder   mhe+te --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_cat_encoder='mhe+te', other_cat_encoder='TargetEncoder', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('------ TargetEncoder   one-hot --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_cat_encoder='one-hot', other_cat_encoder='TargetEncoder', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('------ TargetEncoder   TargetEncoder --------------')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_cat_encoder='TargetEncoder', other_cat_encoder='TargetEncoder', verbose=True)
        assert np.sum(np.isnan(y_train)) == 0
        assert np.sum(np.isnan(y_test)) == 0

        print('============= expecting ValueError =========')
        X_train,y_train,X_test,y_test = prepare_train_test(ds, train_index, test_index, hc_col, hc_group=42, hc_cat_encoder='mhe', other_cat_encoder='TargetEncoder', verbose=True)
    
    # df = pd.DataFrame()
    # df['A'] = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    # df['B'] = pd.Series([1,2,3,4,5,7,8,9],dtype='category')
    # df['C'] = pd.Series(['F', 'M', 'F', 'M', 'F', 'N', 'N', 'F'], dtype='category')
    # df['Y'] = [1,1,1,0,0,1,1,1,0]    
    # print(f'{df.dtypes}')
    # t = TargetEncoder()
    # t.fit(df[['A','B','C']], df['Y'])
    # a = t.transform(df[['A','B','C']])
    # print(f'{a=}')
    
    # X =df[['A','B','C']].to_numpy()
    # t = TargetEncoder()
    # t.fit(X, df['Y'])
    # b = t.transform(X)
    # print(f'{b=}')

    # categorical_transformer = Pipeline(steps=[
    #         ('imputer', SimpleImputer(strategy='most_frequent')),
    #         ('encoder', TargetEncoder()),
    #         #('encoder', OrdinalEncoder())
    #         ])
    # prepro = ColumnTransformer(transformers=[
    #         ('cat', categorical_transformer, ['B','C'])],
    #     remainder = 'drop') 
    # prepro.fit(df,df['Y'])
    # c = pd.DataFrame(prepro.transform(df))
    # print(f'{c=}')

    # col_conf = encoder.ColumnConfig(group_count = 2, idx = 0,
    #                         min_samples_per_cat=1,
    #          insufficient_handler = encoder.CategoryHandler.CAT_TYPICAL,
    #          missing_handler = encoder.CategoryHandler.CAT_TYPICAL, # we already filled missing values, no need for the special group of missing
    #          unseen_handler = encoder.CategoryHandler.CAT_TYPICAL)
    # mhe = encoder.MaxHomoEncoder([col_conf])
    # pip = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='most_frequent')),
    #     ('mhe', mhe),
    #     ('one-hot',OneHotEncoder(handle_unknown='ignore', # unknown value in transform is encoded as zero vector
    #                         # drop='first',          # remove the first category to avoid collinearity
    #                         sparse = False)          # return a dense matrix
    #      )])
    # pip.fit(df[['C']], df['Y'])
    # d = pip.transform(df[['C']])
    # print(f'{d=}')
    

    # df = pd.DataFrame()
    # df['A'] = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    # df['B'] = pd.Series([1,2,3,4,5,6,7,8,9],dtype='category')
    # df['C'] = pd.Series(['F', 'M', 'F', 'M', 'F', 'F', 'N', 'N', 'M'], dtype='category')
    # df['Y'] = [1,1,1,0,0,1,1,1,0]    
    # print(f'{df.dtypes}')
    # df_train = df.loc[:5]
    # df_test = df.loc[6:]
    # print(f'{df_train=}')
    # print(f'{df_test=}')
    # print(f'{df_test["C"].dtype=}')
    
    # t = TargetEncoder()
    # t.fit(df_train[['A','B','C']], df_train['Y'])
    # print(f'{t.mapping["C"]=}')
    # print(f'{t.ordinal_encoder.category_mapping[1]=}')
    # a = t.transform(df_train[['A','B','C']])
    # print(f'{a=}')
    # b = t.transform(df_test[['A','B','C']])
    # print(f'{b=}')
    
    # # Test the need for reset_index
    # df = pd.DataFrame()
    # df['A'] = ['a','b','c','d']
    # df['y'] = [0,1,0,1]
    # train_index = [1,3]
    # df_train = df[['A']].iloc[train_index,:]    # index 1, 3
    # y_train = df['y'].iloc[train_index]         # index 1, 3
    # print(f'{df_train=}')
    # print(f'{y_train=}')
    # x_train = pd.DataFrame(df_train.to_numpy()) # index 从 1 开始连续整数
    # x_train['y'] = y_train                      # 按相同 index 把 x 与 y 拼接， 导致第 1 行 （x.index = 3) 找不到对应的 y
    # print(f'{x_train=}')

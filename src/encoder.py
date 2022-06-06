# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 14:52:49 2021

@author: iwenc

设计理念：
1. 训练集、测试集存储格式
    允许用户用 pandas.DataFrame、 numpy.ndarray 的2维数组或者 python 内建的 list
    中嵌套list的方式提供训练集及测试集；
    程序内部尽量用 pandas.DataFrame 可以借助 pandas 处理表格的能力简化各种计算的逻辑

2. 将一个 categorical variable 的 category 分成4种：
    missing:      数据中没有填写的缺失值，pandas 及 numpy 用 numpy.nan （一个特殊的
                  float64 值） 表示缺失。
    insufficient: 训练集中该类样本很少，比如少于 3 条的
    normal:       训练集中该类样本比较多，比如 >= 3 条的
    unseen:       在测试集中出现，但训练集中没有出现的类
3. 针对 normal category 采用 max homogeneity DP 编码成 J 个组

4. 特殊 category： missing、insufficient、unseen 允许用户通过 CategoryHandler 指定
   如何编码：
    CAT_TYPICAL       # 当作训练集中 normal catogry 中正样本比例最接近于训练集整体正样本比例的类来处理
    CAT_LARGEST       # 当作训练集中 normal catogry 中样本数最多的 category 来处理
    CAT_INSUFFICIENT  # 合并到一个特殊分组 group_cat_insufficient （数据比较少的类形成的组）
    CAT_MISSING       # 合并到一个特殊分组 group_cat_missing （缺少类的样本形成的组）

5. mhe = MaxHomoEncoder(col_conf_list)  # 创建一个编码器并指定，哪些列应该编码，如何编码
   用 ColumnConfig 类说明如何对一个列进行编码，及如何处理特殊种类的 category
    group_count       # 编码以后的最大组数（包括可能需要的1-2个特殊分组）
    name              # 需要编码的列的名称
    idx               # 需要编码的列是第几列，name与idx至少要有一个，如果都有，name优先
    min_samples_per_cat  # 一个类包含多少条数据才算正常来（数据足够）
    insufficient_handler、missing_handler、unseen_handler  # 指定各种类如何处理
    
    先根据 group_count 及是否需要特殊组，确定normal category可以用的分组数 J
        J + 特殊分组数 <= group_count
        J <= 训练集中出现的 normal catgory 数
        
6. mhe.fit(X, y) # 用 X 做训练集，y 为binary向量，训练编码器
   根据用户指定的处理偏好及训练集中的数据可能创建 1 - 2 个需要用到的特殊分组。
   构建一个 pandas.Series 类型的 mapping 来表示每个训练集中见到的类分配到哪个组。
       Index        value
       'He Gang'    2     # group_cat_insufficient
       'Guang Zhou' 0
       'Bei Jing'   0
       nump.nan     3     # group_cat_missing
       'Ji Nan'     1
       'Xi An'      1
   组别用从 0 开始的整数编号，先编码正常的组别，然后依次按需要创建 group_cat_insufficient
   及 group_cat_missing。 如果有 group_cat_missing 分组它一定是最后一组。   

7. mhe.transform(X_t) 根据训练好的编码器及用户指定的特殊类处理偏好进行编码
   


关于表格中的缺失值，numpy.nan 的注意事项：
1. numpy.nan 是 float 类型，所以如果用 numpy.ndarray 则数据类型不能是 int 只能是 float
2. 可以用 numpy.isnan() 来判断，即 numpy.isnan(numpy.nan) 的结果是 True
3. 不能用 == 来比较，即 numpy.nan == numpy.nan 的值是 False
4. 可以出现在 pandas.Index 中，即表格 （DataFrame) 或者列 （Series) 的索引中
   a = pandas.Series([1,2,3], index=[numpy.nan, 'a', 'b'])
5. 可以用 in 来判断 numpy.nan 是否在一个 collection 中，比如：
   (numpy.nan in [3, numpy.nan, 5]) 的值为 True
   (numpy.nan in pandas.Index([numpy.nan, 'a', 'b'])) 的值也是 True
6. numpy.nan放进pandas.Index后拿出来，就于 numpy.nan 不太一样了
   a = [numpy.nan]
   b = pandas.Index([numpy.nan])
   e = b.iloc[0]
   numpy.isnan(e)  # 值为 True
   e in a          # 值为 False
   numpy.nan in a  # 值为 True
7. pandas 中如果一个列有numpy.nan，把它变成 'category' 类型以后. numpy.nan 不是作为
   一个 category
   a = pandas.Series(['GZ', numpy.nan])
   b = a.astype('category')
   b.dtype   # 是一个只包含一个类别 'GZ' 的 categorical 类型
   b.iloc[2] # 还是 numpy.nan 表示缺失
   m = pandas.Series([10,20], index=['GZ', numpy.nan])
   a.map(m)  # 'GZ' 与 numpy.nan 分别被替换成 10 与 20
   b.map(m)  # categorical 中的 'GZ' 被替换成 10, numpy.nan 不是一个类别，没有被替换
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import unittest
import time

from enum import Enum

class CategoryHandler(Enum):
    CAT_TYPICAL = 0       # The category c, P(Y=1|X=c) is most similar to P(Y=1)
    CAT_LARGEST = 1       # The category with most number of samples in training data
    CAT_INSUFFICIENT = 2  # A category with insufficient data
    CAT_MISSING = 3       # missing category


def nan_safe_subseteq(sub, seq):
    '''
    Check if sub after ignoring duplicated items, is a subset of seq after ignoring duplicated items.
    Both sub and seq may contains numpy.nan (where simple comparision fails as numpy.nan != numpy.nan in python)

    Parameters
    ----------
    sub : sequence or pandas.Index
        DESCRIPTION.
    seq : sequence or pandas.Index
        DESCRIPTION.

    Returns
    -------
    bool
        True if sub contains a subset of items of seq; False otherwise.

    '''
    for e in sub:
        # np.nan stored in pd.Index seems changed, even though np.isnan indicates it is still nan
        # need to manually convert it to np.nan when it is taken from pd.Index
        #
        # a = [np.nan]
        # b = pd.Index([np.nan])
        # a[0] in b=True, a[0]=nan
        # b[0]=nan, np.isnan(b[0])=True, b[0] in a=False
        # np.isnan(np.nan)=True, np.nan in a=True
        if isinstance(e, numbers.Number) and np.isnan(e): 
             e = np.nan
        if (e not in seq):
            return False
    return True

class ColumnConfig:
    def __init__(self, group_count, name=None, idx=None, 
                 min_samples_per_cat=3,
                 insufficient_handler = CategoryHandler.CAT_INSUFFICIENT,
                 missing_handler = CategoryHandler.CAT_MISSING,
                 unseen_handler = CategoryHandler.CAT_INSUFFICIENT):
        '''
        A configuration specifies how to encode categories in a column.
        A categorical variable is stored as a column in a pandas.DataFrame
        (or 2D numpy arrays that can be convereted into a pandas.DataFrame)                 

        Parameters
        ----------
        group_count : int
            Total number of groups. This includes two optimal special groups:
            1. A special group for categories with insufficient training data. This group is created if
                a. there are categories with number of samples less than min_samples_per_cat
                   in training data and insufficient_handler = CategoryHandler.CAT_INSUFFICIENT, or
                b. missing_handler=CategoryHandler.CAT_INSUFFICIENT, or
                c. unseen_handler=CategoryHandler.CAT_INSUFFICIENT
            2. A special group for missing category. That is a gorup is created if
                a. there are categories with number of samples less than min_samples_per_cat
                   in training data and insufficient_handler = CategoryHandler.CAT_MISSING, or
                b. missing_handler=CategoryHandler.CAT_MISSING, or
                c. unseen_handler=CategoryHandler.CAT_MISSING
        name : str, optional
            column name, that store a categorical variable. The default is None.
        idx : int, optional
            index of column that store a categorical variable. The default is None.
            At least one of name or idx must be specififed. If both are specified
            use column name to compute and override idx
        min_samples_per_cat : int, optional
            The minimum number of samples in a category for it be considered a
            normal category. The default is 3.
            When a category contains training samples less than this number,
            it is considered as a category with insufficient data and need
            special treatment as indicated by insufficient_handler during fit()
        insufficient_handler : CategoryHandler, optional
            Specifies how category with insufficient data should be handled.
            The default is CategoryHandler.CAT_INSUFFICIENT.
        missing_handler : CategoryHandler, optional
            Specifies how missing category should be handled.
            The default is CategoryHandler.CAT_MISSING.
        unseen_handler : CategoryHandler, optional
            Specifies how unseen categories should be handled.
            The default is CategoryHandler.CAT_INSUFFICIENT.

        Returns
        -------
        None.

        '''
        self.group_count = group_count
        self.name = name
        self.idx = idx
        self.min_samples_per_cat = min_samples_per_cat
        self.insufficient_handler = insufficient_handler
        self.missing_handler = missing_handler
        self.unseen_handler = unseen_handler
        
        if (name is None) and (idx is None):
            raise ValueError('both name and idx is None, at least one should be specified')
        
        if type(min_samples_per_cat) != int and type(min_samples_per_cat) != float:
            raise ValueError(f'{min_samples_per_cat=} must be a number (either int or float)')       
        if min_samples_per_cat < 0:
            raise ValueError(f'{min_samples_per_cat=} must be zero or positive')

        if not isinstance(insufficient_handler,CategoryHandler):
            raise ValueError(f'{insufficient_handler=} is not an instance of UnknownCategoryHandler')

        if not isinstance(missing_handler,CategoryHandler):
            raise ValueError(f'{missing_handler=} is not an instance of UnknownCategoryHandler')
        
        if not isinstance(unseen_handler,CategoryHandler):
            raise ValueError(f'{unseen_handler=} is not an instance of UnknownCategoryHandler')

    
           


class TestColumnConfig(unittest.TestCase):
    def setUp(self):
        pass

    def test___init__(self):
        self.assertRaises(ValueError, ColumnConfig, 5)
        
        conf = ColumnConfig(5, name='City')
        self.assertEqual(5, conf.group_count)
        conf = ColumnConfig(5, idx=3)
        self.assertEqual(3, conf.idx)
        
        self.assertRaises(ValueError, ColumnConfig, 5, idx=3, min_samples_per_cat=-1)
        self.assertRaises(ValueError, ColumnConfig, 5, idx=3, min_samples_per_cat=None)
        self.assertRaises(ValueError, ColumnConfig, 5, idx=3, unseen_handler=None)

        conf = ColumnConfig(6, idx=2, unseen_handler=CategoryHandler.CAT_LARGEST)
        self.assertEqual(6, conf.group_count)
        self.assertEqual(2, conf.idx)
        self.assertIsNone(conf.name)
        self.assertEqual(CategoryHandler.CAT_LARGEST, conf.unseen_handler)
    

def validate_binary_vector(y, n):
    '''
    Check if y is a binary vector with length n.

    Parameters
    ----------
    y : pandas.Series, convertable to pandas.Series
        If y is pandas.DataFrame with only one column, its first column is taken
        If y is not a pandas.Series, such as a list, dict. It is first converted into pandas.Series
        If y is not numeric, it is converted to numeric.
        After conversion it must contains only 1 and 0 (allow 1.0 and 0.0)
    n : expected length of vector
    Raises
    ------
    ValueError
        1. if y contains elements not equal to 1 and 0
        2. if length of y is not n

    Returns
    -------
    pandas.Series of numbers representing the binary vector y.

    '''
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 1:
            s_y = y.iloc[:,0] # get first column
        else:
            raise ValueError(f'y is pandas.DataFrame, it should be either a pandas.Series, list, 1d array, {y=}')
    elif isinstance(y, pd.Series):
        s_y = y
    else:
        s_y = pd.Series(y)    

    if s_y.size != n:
        raise ValueError(f'number of elements in y: {s_y.size} != {n=}')

    if not pd.api.types.is_numeric_dtype(s_y): # convert y to numbers if not
        s_y = pd.to_numeric(s_y)
    flag = (s_y==1) | (s_y==0)
    if flag.sum() != s_y.size:  # check y are either 1 or 0, 1.0 and 0.0 is also okay
        raise ValueError(f'y contains value other than 0 and 1, {y=}')
    
    return s_y

from pandas.testing import assert_series_equal
class TestValidate_binary_vector(unittest.TestCase):
    def setUp(self):
        pass

    def test_validate_binary_vector(self):
        y = [1,3,2]
        self.assertRaises(ValueError, validate_binary_vector, y, 3)
        
        s_y = validate_binary_vector([1,0,1.0,0.0], 4)
        assert_series_equal(pd.Series([1,0,1.0,0.0]), s_y)
        
        y = pd.DataFrame({'City':[1.0,0]})
        s_y = validate_binary_vector(y, 2)
        assert_series_equal(y['City'], s_y)

        y = np.array([1.0,0])    
        s_y = validate_binary_vector(y, 2)
        assert_series_equal(pd.Series(y), s_y)
        self.assertRaises(ValueError, validate_binary_vector, y, 3)
        
        s_y = validate_binary_vector(0, 1)
        assert_series_equal(pd.Series(0), s_y)

        self.assertRaises(ValueError, validate_binary_vector, 5, 1)
        
        y = [True, False, True]
        s_y = validate_binary_vector(y, 3)
        assert_series_equal(pd.Series(y), s_y)




def dp(cat_N,cat_P,J):
    '''
    Dynamic programming formulation for partitioning
    I categories: 0, 1, ..., I-1 into J groups: 0, 1, ..., J-1
    So that total homogeneity is maximized.

    Where categories are in ascending order of cat_P.iloc[i] / cat_N.iloc[i]

    Parameters
    ----------
    cat_N : pandas.Series
        cat_N.iloc[i] > 0, number of samples in i-th categories, i=0, ..., I-1
    cat_P : pandas.Series
        cat_N.iloc[i] <= N[i], P[i] >= 0, number of positive samples in i-th categories, i=0, ..., I-1
    J : int
        J>2, J<len(N). Number of groups

    Returns
    -------
    mapping : pandas.Series
        mapping.iloc[i] is the group assigned to category i
    cats_per_group: np.nadrray 1D
        cats_per_group[j] the number of categories in group j

    '''
    assert cat_N.size == cat_P.size    
    N = cat_N.to_numpy()
    P = cat_P.to_numpy()

    for i in range(len(N)):
        assert N[0] > 0
        assert P[0] >= 0    
    p = P/N
    for i in range(1,len(p)):
        assert p[i] >= p[i-1]

    # def homo(N,P):
    #     if N == 0:
    #         return 0
    #     return - 2 * P * (1-P/N)        # print(f'{N=}')    
    # print(f'{P=}')    
    # for (n,p) in zip(N,P):
    #     print(f'{homo(n,p)=}')
    
    I = N.shape[0]          # number of categories
    f = np.zeros((J+1,I+1)) # f[j,i] the best objective of partition the firt i categories: 0,...,i-1 into j groups
    d = np.zeros((J+1,I+1), dtype='int64')     
    f[:,:] = -np.inf
    f[0,0] = 0              # base case, when j = 0, no data ==> homo = 0   
    
    # N_cum[i] = N[0] + ... + N[i-1], the number of samples in first i categories
    N_cum = np.zeros((I+1,))    
    N_cum[1:] = N.cumsum()
    # P_cum[i] = P[0] + ... + P[i-1], the number of positive samples in first i categories
    P_cum = np.zeros((I+1,))    
    P_cum[1:] = P.cumsum()
    
    # print(f'{N_cum=}')
    # print(f'{P_cum=}')
    
    for j in range(1,J+1):   # j = 1, 2, ..., J
        # print(f'{j=}')
        f[j,0:j] = -np.inf   # i = 0, ..., j-1, i < j, impossible to have one category per group
        # print(f'{f[j-1]=}')
        # print(f'{f[j]=}')
        # print(f'{d[j-1]=}')
        # print(f'{d[j]=}')
        for i in range(j,I+1):  # i = j, ..., I
            # print(f'  {i=}')
            max_homo = -np.inf
            max_k = 0
            for k in range(1,i-j+2):    # max k: i-k = j-1 ==> k = i-j+1
                # print(f'    {k=}')
                start_i = i-k           # i-k items: 0, ..., i-k-1 in first j-1 groups 
                                        # the last group: i-k, ..., i-1 
                # print(f'    {start_i=}')
                N_last = N_cum[i] - N_cum[start_i]
                P_last = P_cum[i] - P_cum[start_i]
                # print(f'    {N_last=}')
                # print(f'    {P_last=}')
                assert N_last > 0
                homo_last = - 2 * P_last * (1-P_last/N_last)
                # print(f'    {homo_last=}')
                obj = f[j-1,i-k] + homo_last
                # print(f'    {obj=}')
                if obj > max_homo:
                    max_homo = obj
                    max_k = k
            f[j,i] = max_homo
            d[j,i] = max_k

    # print(f'{f=}')
    # print(f'{d=}')
    
    cats_per_group = np.zeros((J+1), dtype='int64')
    i = I
    for j in range(J,0,-1):
        max_k = d[j,i]
        cats_per_group[j] = max_k
        i -= max_k
    # print(f'{cats_per_group=}')
    
    cats_cum = cats_per_group.cumsum()
    mapping = pd.Series(dtype='int64', index=cat_N.index)
    for j in range(0,J):
        start_i = cats_cum[j]
        end_i = cats_cum[j+1]
        # print(f'group: {j+1}, {start_i=}, {end_i=}')
        for i in range(start_i,end_i):
            mapping.iloc[i] = j

    # print(f'{mapping=}')    
    
    return mapping, cats_per_group[1:]

class Test_dp(unittest.TestCase):
    def setUp(self):
        pass

    def test_dp_01(self):
        cat_N = pd.Series([3,2], index=['B', 'A'])
        cat_P = pd.Series([1,1], index=['B', 'A'])    
        mapping,cats_per_group = dp(cat_N,cat_P,1)
        assert_series_equal(pd.Series([0,0], index=['B', 'A']), mapping)
        np.testing.assert_array_equal(np.array([2]), cats_per_group)

    def test_dp_02(self):
        cat_N = pd.Series([3,2,3], index=['B', 'A', np.nan])
        cat_P = pd.Series([1,1,2], index=['B', 'A', np.nan])    
        mapping,cats_per_group = dp(cat_N,cat_P,2)
        assert_series_equal(pd.Series([0,0,1], index=['B', 'A', np.nan]), mapping)
        np.testing.assert_array_equal(np.array([2,1]), cats_per_group)


def _fit_one_column(x, y, col_conf):
    '''
    Groupy y by unique values in x (including numpy.nan, numpy.inf, -numpy.inf)

    Set the following attributes on col_conf
        total_group_count_      # total number of groups including the two optional special groups
    Set the following, if insufficient, missing, or unseen category handler need CAT_INSUFFICIENT:
        group_cat_insufficient_ # the group number for CAT_INSUFFICIENT
    Set the following, if insufficient, missing, or unseen category handler need CAT_MISSING:
        group_cat_missing_      # the group number for CAT_MISSING
    Set the following, if insufficient, missing, or unseen category handler need CAT_TYPICAL:
        cat_typical_            # the typical category in training data, 
        group_cat_typical_      # the group assigned to the typical category
    Set the following, if insufficient, missing, or unseen category handler need CAT_LARGIEST:
        cat_largiest_           # the largest category in training data
        group_cat_largiest_     # the group assigned to largest category

    Parameters
    ----------
    x : pandas.Series
        Each unique value of x is treated as a category
    y : pandas.Series
        A binary vector with same elements as x
    col_conf : ColumnConfig
        A configuration indicating how to encode the column x
        The categories attributes is set to distinct values in x if it is None
    Returns
    -------
    mapping: pandas.Series
        The index representing the category and value represents its group
    '''
    assert y.size == x.size
    
    tbl = y.groupby(x, dropna=False).agg(['sum', 'count'])
    tbl.columns = ['sum', 'count']
    tbl['p'] = tbl['sum'] / tbl['count']
    tbl = tbl.sort_values(by='p')
    tbl['group'] = np.nan


    # Split categories into 3 types:
    #     1. missing category, where label is numpy.nan
    #     2. normal category, label is not numpy.nan and has at least min_samples_per_cat samples
    #     3. insufficient category, label is not numpy.nan and has less than min_samples_per_cat samples
    # Set the following attributes:
    #     _flag_normal: indicating whether a category is normal category, count[_flag_normal] select the sample counts for normal categories
    #     _flag_insufficient: indicating whether a category is insufficient category
    #     _has_missing: bool, whether there is missing category
    #     _has_insufficient: bool, whether there are insufficient categories
    categories = tbl['count'].index
    flag_missing = categories.isna()
    flag_has_value = ~flag_missing
    flag_normal = flag_has_value & (tbl['count']>=col_conf.min_samples_per_cat)
    flag_insufficient = flag_has_value & (tbl['count']<col_conf.min_samples_per_cat)
    
    has_insufficient_cat = flag_insufficient.sum() > 0
    has_insufficient_group = (has_insufficient_cat and
            col_conf.insufficient_handler == CategoryHandler.CAT_INSUFFICIENT
        or col_conf.missing_handler == CategoryHandler.CAT_INSUFFICIENT
        or col_conf.unseen_handler == CategoryHandler.CAT_INSUFFICIENT)

    has_missing_group = (has_insufficient_cat and 
            col_conf.insufficient_handler == CategoryHandler.CAT_MISSING
        or col_conf.missing_handler == CategoryHandler.CAT_MISSING
        or col_conf.unseen_handler == CategoryHandler.CAT_MISSING)

    mhe_group_count = col_conf.group_count
    if has_insufficient_group:
        mhe_group_count -= 1
    if has_missing_group:
        mhe_group_count -= 1
    norm_cat_count = flag_normal.sum()
    if norm_cat_count < mhe_group_count:
        mhe_group_count = norm_cat_count
        
    col_conf.total_group_count_ = mhe_group_count
    if has_insufficient_group:
        col_conf.group_cat_insufficient_ = col_conf.total_group_count_
        col_conf.total_group_count_ += 1
    if has_missing_group:
        col_conf.group_cat_missing_ = col_conf.total_group_count_
        col_conf.total_group_count_ += 1

    if (col_conf.insufficient_handler == CategoryHandler.CAT_TYPICAL
        or col_conf.missing_handler == CategoryHandler.CAT_TYPICAL
        or col_conf.unseen_handler == CategoryHandler.CAT_TYPICAL):
        if norm_cat_count > 0:
            global_N = tbl['count'].sum()
            if global_N == 0:
                global_p = 0
            else:
                global_p = tbl['sum'].sum() / global_N
            tbl['dist-to-mean'] = -abs(tbl['p']-global_p)
            col_conf.cat_typical_ = tbl.loc[flag_normal,'dist-to-mean'].idxmax()
        else:
            raise ValueError(f'Typical column is needed, but there is no category with sample count >= {col_conf.min_samples_per_cat=}')

    if (col_conf.insufficient_handler == CategoryHandler.CAT_LARGEST
        or col_conf.missing_handler == CategoryHandler.CAT_LARGEST
        or col_conf.unseen_handler == CategoryHandler.CAT_LARGEST):
        if norm_cat_count > 0:
            col_conf.cat_largest_ = tbl.loc[flag_normal,'count'].idxmax()
        else:
            raise ValueError(f'Largest column is needed, but there is no category with sample count >= {col_conf.min_samples_per_cat=}')

    # there are normal categories, but mhe_group_count_ is too small
    if mhe_group_count < 1:
        if norm_cat_count > 0:
            raise ValueError(f'{col_conf.mhe_group_count_=}, try to increase {col_conf.group_count=} for column {col_conf.idx=}')

    if mhe_group_count > 0:
        cat_N = tbl['count'][flag_normal]
        cat_P = tbl['count'][flag_normal]
    
        mapping,_ = dp(cat_N, cat_P, mhe_group_count)    
        tbl.loc[flag_normal,'group'] = mapping    # replace selected rows in 'group' column by mapping
        if hasattr(col_conf, 'cat_typical_'):
            col_conf.group_cat_typical_ = mapping[col_conf.cat_typical_]
        else:
            col_conf.group_cat_typical_ = None
        if hasattr(col_conf, 'cat_largest_'):
            col_conf.group_cat_largest_ = mapping[col_conf.cat_largest_]
        else:
            col_conf.group_cat_largest_ = None

    if has_insufficient_cat:
        if col_conf.insufficient_handler == CategoryHandler.CAT_TYPICAL:
            tbl.loc[flag_insufficient,'group'] = col_conf.group_cat_typical_
        elif col_conf.insufficient_handler == CategoryHandler.CAT_LARGEST:
            tbl.loc[flag_insufficient,'group'] = col_conf.group_cat_largest_
        elif col_conf.insufficient_handler == CategoryHandler.CAT_INSUFFICIENT:
            tbl.loc[flag_insufficient,'group'] = col_conf.group_cat_insufficient_
        else: # col_conf.insufficient_handler == CategoryHandler.CAT_MISSING:
            tbl.loc[flag_insufficient,'group'] = col_conf.group_cat_missing_

    if flag_missing.sum() > 0:
        if col_conf.missing_handler == CategoryHandler.CAT_TYPICAL:
            tbl.loc[flag_missing,'group'] = col_conf.group_cat_typical_
        elif col_conf.missing_handler == CategoryHandler.CAT_LARGEST:
            tbl.loc[flag_missing,'group'] = col_conf.group_cat_largest_
        elif col_conf.missing_handler == CategoryHandler.CAT_INSUFFICIENT:
            tbl.loc[flag_missing,'group'] = col_conf.group_cat_insufficient_
        else: # col_conf.missing_handler == CategoryHandler.CAT_MISSING:
            tbl.loc[flag_missing,'group'] = col_conf.group_cat_missing_

    return tbl['group']

import numbers
class MyTestCase(unittest.TestCase):
    def assertSubsetEq(self, sub, seq):
        if not nan_safe_subseteq(sub, seq):
            raise AssertionError(f"{sub=} is not a subset of {seq=}")
            
    def assertSameSet(self, a, b):
        self.assertSubsetEq(a,b)
        self.assertSubsetEq(b,a)

class Test_encode_one_column(MyTestCase):
    def setUp(self):
        pass

    # def test_encode_one_column_01(self):
    #     col_conf = ColumnConfig(4,idx=0)
    #     x = pd.Series([np.inf, 'b', 'b', 'b', 'c', 'c', np.inf, np.nan, np.inf])
    #     y = pd.Series([0,1,0,0,1,1,0,0,1])
    #     _fit_one_column(x,y,col_conf)

    def test_encode_one_column_02(self):
        col_conf = ColumnConfig(4,idx=0,missing_handler=CategoryHandler.CAT_LARGEST, unseen_handler=CategoryHandler.CAT_TYPICAL)
        x = pd.Series([np.inf, 'b', 'b', 'b', 'c', 'c', np.inf, np.nan, np.inf, 'b'])
        y = pd.Series([0,1,0,0,1,1,0,0,1,0])
        _fit_one_column(x,y,col_conf)
        self.assertEqual('b', col_conf.cat_largest_)
        self.assertEqual(np.inf, col_conf.cat_typical_)
        self.assertEqual(3, col_conf.total_group_count_)
        self.assertEqual(0, col_conf.group_cat_largest_)
        self.assertEqual(1, col_conf.group_cat_typical_)

    # def test_encode_one_column_02(self):
    #     col_conf = ColumnConfig(2,idx=0)
    #     x = pd.Series(['a', 'b', 'a', 7])
    #     y = pd.Series([0,1,1,0])
    #     _fit_one_column(x,y,col_conf)
    #     self.assertCountEqual(['b', 'a', 7], col_conf.categories_) # cannot handle nan correctly



        
class MaxHomoEncoder(TransformerMixin, BaseEstimator):
    
    def __init__(self, col_config_list, mapping_list=None):
        '''
        Create a maximum homogeneity encoder for several columns.

        Parameters
        ----------
        col_config_list : List[ColumnConfig]
            A list of configuration for each column to be encoded. 
                  
        mapping_list : List[pd.Series], optimal
            This is the result of fit(). For each column, a pd.Series is
            used to define the map from a category to a group, where index
            is category and value is its group.            
            The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if isinstance(col_config_list, list):
            if len(col_config_list) < 1:
                raise ValueError(f'{col_config_list=} is empty')
            self.col_config_list = col_config_list
        else:
            self.col_config_list = [col_config_list]
        
        for idx,con_config in enumerate(self.col_config_list):
            if not isinstance(con_config, ColumnConfig):
                raise ValueError(f'The {idx}-th configuration in col_config_list is not an instance of ColumnConfig, {col_config_list=}')

        self.mapping_list = mapping_list
        
        if mapping_list is not None and len(mapping_list) != len(self.col_config_iter):
            raise ValueError(f'Number of columns in mapping_list: {len(mapping_list)} != number of configurations in col_config_list: {len(self.self.col_config_list)}')

    

    def fit(self, X, y):
        '''
        Use training data X to find the mapping from category to group for each column.

        Call _fit_one_column for each ColumnConfig obj in self.col_config_list
        Which will set a few attributes on ColumnConfig as the result of fitting   
        
        Parameters
        ----------
        X : padas.DtaFrame, numpy.ndarray, list()
            A data frame, 2D numpy array that describe training data
            Every row is a training sample, every column is a variable.
            For numpy.ndarray, X[:,i] is the i-th column
        y : pandas.Series, numpy.ndarray, list()
            A Series, 1D numpy array describe a binary vector.
            That is every entry of the vector is either 0 or 1 (can be 0.0 and 1.0)

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            self for chained call.

        '''
        if isinstance(X, pd.DataFrame):
            df_X = X
        else:
            df_X = pd.DataFrame(X)
        
        s_y = validate_binary_vector(y, df_X.shape[0])
        
        for conf_idx, col_conf in enumerate(self.col_config_list):
            if col_conf.name is not None:
                try:
                    idx = df_X.columns.get_loc(col_conf.name)
                except KeyError:
                    raise ValueError(f'{col_conf.name=} is not found in columns: {df_X.columns=} in {conf_idx}-th column configuration')
                # if col_conf.idx is None:
                #     col_conf.idx = idx
                # elif col_conf.idx != idx:
                #     raise ValueError(f'{col_conf.name=} corresponds to {idx}-th column != {col_conf.idx=}, in {conf_idx}-th column configuration')
                col_conf.idx = idx
                
                
        self.fit_dp_times_ = []    
        self.mapping_list = []
        for conf_idx, col_conf in enumerate(self.col_config_list):
            s_x = df_X.iloc[:,col_conf.idx]  # get the column from X
            start = time.time()
            mapping = _fit_one_column(s_x, s_y, col_conf)
            end = time.time()
            self.mapping_list.append(mapping)
            self.fit_dp_times_.append(end-start)
            
        return self


    def transform(self, X, y=None):
        '''
        Transform categorical column by applying category to group mapping.
        '''
        if isinstance(X, pd.DataFrame):
            df_X = X
            X_copy = X.copy()
        elif isinstance(X, np.ndarray):
            df_X = pd.DataFrame(X)
            X_copy = X.copy()
        else:
            df_X = pd.DataFrame(X)
            X_copy = [i.copy() for i in X]
        
        
        for conf_idx, col_conf in enumerate(self.col_config_list):
            s_x = df_X.iloc[:,col_conf.idx]  # get the column from X
            mapping = self.mapping_list[conf_idx]
            cats = s_x.value_counts(dropna=False).index
            extended_mapping = mapping.reindex(index=cats)
            # print(f'{mapping=}')
            # print(f'{cats=}')
            # print(f'{extended_mapping=}')

            flag_missing = extended_mapping.index.isna()
            # print(f'{flag_missing=}')
            missing_count = flag_missing.sum()
            # print(f'{missing_count=}')
            group_for_missing = None
            if missing_count > 0: # there are missing in test set
                assert missing_count == 1
                # if there is no missing in traing set, the group value correpsoinding to np.nan is not set
                if extended_mapping[flag_missing].isna().sum() > 0: 
                    if col_conf.missing_handler == CategoryHandler.CAT_TYPICAL:
                        group_for_missing = col_conf.group_cat_typical_
                    elif col_conf.missing_handler == CategoryHandler.CAT_LARGEST:
                        group_for_missing = col_conf.group_cat_largest_
                    elif col_conf.missing_handler == CategoryHandler.CAT_INSUFFICIENT:
                        group_for_missing = col_conf.group_cat_insufficient_
                    else: # col_conf.missing_handler == CategoryHandler.CAT_MISSING:
                        group_for_missing = col_conf.group_cat_missing_
                    extended_mapping.loc[flag_missing] = group_for_missing
            # print(f'{group_for_missing=}')
            
            # category appear in test but not training, its group in extended_map is np.nan
            flag_unseen = extended_mapping.isna()
            if flag_unseen.sum() > 0: # there are unseen data
                if col_conf.unseen_handler == CategoryHandler.CAT_TYPICAL:
                    extended_mapping.loc[flag_unseen] = col_conf.group_cat_typical_
                elif col_conf.unseen_handler == CategoryHandler.CAT_LARGEST:
                    extended_mapping.loc[flag_unseen] = col_conf.group_cat_largest_
                elif col_conf.unseen_handler == CategoryHandler.CAT_INSUFFICIENT:
                    extended_mapping.loc[flag_unseen] = col_conf.group_cat_insufficient_
                else: # col_conf.unseen_handler == CategoryHandler.CAT_MISSING:
                    extended_mapping.loc[flag_unseen] = col_conf.group_cat_missing_
        
            # if s_x is categorical, new_x will be category_cal
            # when manually map nan to group_for_missing, we may fail because
            # group_for_missing is not a category yet
            # convert group to numerics will resolve this problem
            new_x = pd.to_numeric(s_x.map(extended_mapping))  
            # print(f'{new_x[new_x.isna()]=}')
            # print(f'{new_x.dtype=}')
            if missing_count > 0: # map fail to replace np.nan to its corresponding value
                new_x[new_x.isna()] = group_for_missing
            assert new_x.isna().sum() == 0
            new_x = new_x.astype('int64')
            
            # print('=============')
            # print(f'{s_x[s_x.isna()]=}')
            # print('======xxx=======')
            # print(f'{new_x[new_x.isna()]=}')
            if isinstance(X, pd.DataFrame):
                pd.set_option('mode.chained_assignment','raise')
                X_copy.iloc[:,col_conf.idx] = new_x
                # X.loc[:,col_conf.idx] = new_x
            elif isinstance(X, np.ndarray):
                X_copy[:,col_conf.idx] = new_x.to_numpy()
            else: # handle list
                for (idx,row) in enumerate(X_copy):
                    row[col_conf.idx] = new_x[idx]
            
        return X_copy


class TestMaxHomoEncoder(unittest.TestCase):
    def setUp(self):
        pass

    def test___init__(self):
        pass

    def test_fit_01(self):
        col_conf = ColumnConfig(4,idx=0,missing_handler=CategoryHandler.CAT_LARGEST, unseen_handler=CategoryHandler.CAT_TYPICAL)
        x = pd.Series([np.inf, 'b', 'b', 'b', 'c', 'c', np.inf, np.nan, np.inf, 'b'])
        df_train = pd.DataFrame({'city':x})
        y = pd.Series([0,1,0,0,1,1,0,0,1,0])
        
        mhe = MaxHomoEncoder([col_conf])
        mhe.fit(df_train,y)
        self.assertEqual('b', col_conf.cat_largest_)
        self.assertEqual(np.inf, col_conf.cat_typical_)
        self.assertEqual(3, col_conf.total_group_count_)
        self.assertEqual(0, col_conf.group_cat_largest_)
        self.assertEqual(1, col_conf.group_cat_typical_)
    
    def test_transform_01(self):
        col_conf = ColumnConfig(4,idx=0,missing_handler=CategoryHandler.CAT_LARGEST, unseen_handler=CategoryHandler.CAT_TYPICAL)
        x = pd.Series([np.inf, 'b', 'b', 'b', 'c', 'c', np.inf, np.nan, np.inf, 'b'])
        df_train = pd.DataFrame({'city':x})
        y = pd.Series([0,1,0,0,1,1,0,0,1,0])
        
        mhe = MaxHomoEncoder([col_conf])
        mhe.fit(df_train,y)
        self.assertEqual('b', col_conf.cat_largest_)
        self.assertEqual(np.inf, col_conf.cat_typical_)
        self.assertEqual(3, col_conf.total_group_count_)
        self.assertEqual(0, col_conf.group_cat_largest_)
        self.assertEqual(1, col_conf.group_cat_typical_)
        
        df_test = pd.DataFrame({'city': ['b',np.inf,'d','c',np.nan]})
        trans_test = mhe.transform(df_test)
        result = pd.Series([0, 1, 1, 2, 0], name='city')
        assert_series_equal(result, trans_test['city'])
        self.assertEqual('b', df_test.loc[0,'city'])

    def test_fit_02(self):
        col_conf = ColumnConfig(4,idx=0,missing_handler=CategoryHandler.CAT_LARGEST, unseen_handler=CategoryHandler.CAT_TYPICAL)
        X = np.array([[np.inf], [20], [20], [20], [30], [30], [np.inf], [np.nan], [np.inf], [20]])
        y = np.array([0,1,0,0,1,1,0,0,1,0])
        
        mhe = MaxHomoEncoder([col_conf])
        mhe.fit(X,y)
        self.assertEqual(20, col_conf.cat_largest_)
        self.assertEqual(np.inf, col_conf.cat_typical_)
        self.assertEqual(3, col_conf.total_group_count_)
        self.assertEqual(0, col_conf.group_cat_largest_)
        self.assertEqual(1, col_conf.group_cat_typical_)
        
    def test_transform_02(self):
        col_conf = ColumnConfig(4,idx=0,missing_handler=CategoryHandler.CAT_LARGEST, unseen_handler=CategoryHandler.CAT_TYPICAL)
        X = np.array([[np.inf], [20], [20], [20], [30], [30], [np.inf], [np.nan], [np.inf], [20]])
        y = np.array([0,1,0,0,1,1,0,0,1,0])
        
        mhe = MaxHomoEncoder([col_conf])
        mhe.fit(X,y)

        test = np.array([[20],
                [np.inf],
                [40],
                [30],
                [np.nan]])
        trans_test = mhe.transform(test)
        result = np.array([[0.0], [1.0], [1.0], [2.0], [0.0]])
        np.testing.assert_array_equal(result,trans_test)
        self.assertEqual(20, test[0][0])

    def test_transform_03(self):
        col_conf = ColumnConfig(4,idx=0,missing_handler=CategoryHandler.CAT_LARGEST, unseen_handler=CategoryHandler.CAT_TYPICAL)
        x = pd.Series([np.inf, 'b', 'b', 'b', 'c', 'c', np.inf, np.nan, np.inf, 'b'])
        df_train = pd.DataFrame({'city':x})
        y = pd.Series([0,1,0,0,1,1,0,0,1,0])
        
        mhe = MaxHomoEncoder([col_conf])
        mhe.fit(df_train,y)

        test = [['b'],
                [np.inf],
                ['d'],
                ['c'],
                [np.nan]]
        trans_test = mhe.transform(test)
        result = [[0.0], [1.0], [1.0], [2.0], [0.0]]
        self.assertSequenceEqual(result, trans_test)
        self.assertEqual('b', test[0][0])


def simple_example():
    X = pd.Series(['a',
                          'b','b','b','b','b','b',
                          'c','c','c','c','c','c','c','c','c','c',
                          'd','d','d','d','d','d'])
    y = pd.Series([0,
                         0,1,0,1,0,1,
                         1,1,1,1,1,1,0,0,1,1,
                         1,1,1,1,1,1])
    
    col_conf = ColumnConfig(3,idx=0,missing_handler=CategoryHandler.CAT_LARGEST, unseen_handler=CategoryHandler.CAT_TYPICAL)
    g = _fit_one_column(X,y,col_conf)
    df = pd.DataFrame({'x':X, 'y':y})
    print("-------- a simple binary classification example ------")
    print(df)
    df_agg = df.groupby('x').agg({'y' : ['count', 'mean']})
    print("-------- summary of data ------------------------------")
    print(df_agg)
    print("-------- MHE encoding of each category ----------------")
    print(g)

if __name__ == "__main__":

    simple_example()
        
    # unittest.main()    

    # # Test encoding on Adult set
    # import datasets
    # ds = datasets.load_one_set('Adult')
    # hc_col = ds.get_largest_card_cat_var()
    
    # from sklearn.model_selection import ShuffleSplit
    # rs = ShuffleSplit(n_splits=1, test_size=.30, random_state=0)
    # for train_index, test_index in rs.split(ds.X):
    #     for J in [10]:
    #         col_conf = ColumnConfig(J,name=hc_col)
    #         mhe = MaxHomoEncoder([col_conf])
            
    #         X_train = ds.X.iloc[train_index,:]
    #         y_train = ds.y.iloc[train_index]
    #         X_test = ds.X.iloc[test_index,:]
    #         y_test = ds.y.iloc[test_index]
    
    #         mhe.fit(X_train, y_train)
    #         X_train_trans = mhe.transform(X_train)
    #         X_test_trans = mhe.transform(X_test) 
    #         test_hc_col = X_test_trans[hc_col]
    #         print(f'{test_hc_col.isna().sum()=}')
    #         print(f'{col_conf.total_group_count_=}')
    #         print(f'{col_conf.group_cat_insufficient_=}')
    #         print(f'{col_conf.group_cat_missing_=}')
            
    #         vc_train = X_train[hc_col].value_counts()
    #         print(f'{vc_train[-5:]=}')
    #         vc_train_mhe = X_train_trans[hc_col].value_counts()
    #         print(f"{vc_train_mhe=}")
    #         print(f"{mhe.fit_dp_times_=}")
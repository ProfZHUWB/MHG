# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:39:46 2022

@author: iwenc
"""

import matplotlib.pyplot as plt
import pandas as pd
from exp2_mhe_vs_pca import data_hc_Jlst
import datasets
import os

data_description_file = '../dataset/data-descriptions.xlsx'
#detail_file = '../results/exp2_mhe_vs_pca_details.xlsx'
exp_result_dir = '../results/exp22_04_19'
dir_name = '../results/exp2_mhe_vs_pca' # output dir
detailed_exp_results_fname = dir_name+"/exp2_results_details_relative.csv"

os.makedirs(dir_name, exist_ok=True)
data_summary_file = os.path.join(dir_name,'data_summary.csv')

algo_short_name_dict = {
        'NeuralNet': 'MLP',
        'RBFSVM': 'SVM'
    }

ds_short_name_dict = {
        'AutoLoan': 'Loan',
        'EmployeeSalaries': 'Employ',
        'H1BVisa': 'Visa',
        'RoadSafety': 'Road',
        'TrafficViolations': 'Traffic'
    }


def create_data_summary():
    tbl_stats = pd.DataFrame(columns=['dataset'])

    # use ds_name as index to create table
    for (ds_name,hc_col,JLst) in data_hc_Jlst:
        ds = datasets.load_one_set(ds_name)
        ds_short_name = ds_short_name_dict.get(ds_name,ds_name)
        tbl_stats.loc[ds.base_name,'dataset'] = ds_short_name
        tbl_stats.loc[ds.base_name,'hc_col'] = hc_col
        for col in ds.stats:
            tbl_stats.loc[ds.base_name,col] = ds.stats.loc[hc_col,col]
        p_samples = ds.y.sum()
        tbl_stats.loc[ds.base_name,'p_samples'] = p_samples
        tbl_stats.loc[ds.base_name,'tot_samples'] = len(ds.y)       

    for (ds_name,hc_col,JLst) in data_hc_Jlst:
        ds = datasets.load_one_set(ds_name, no_sampling=True)
        p_count = ds.y.sum()
        tbl_stats.loc[ds.base_name,'p_count'] = p_count
        tbl_stats.loc[ds.base_name,'N'] = ds.X.shape[0]
        tbl_stats.loc[ds.base_name,'l'] = ds.X.shape[1]

    tbl_stats.reset_index(inplace=True)  # convert index to a column with name 'index'
    tbl_stats.rename(columns={'index':'ds_name'},inplace=True) # 'index' --> 'ds_name'
    tbl_stats.to_csv(data_summary_file, index=False)
    
    return tbl_stats


def create_data_summary_brief():
    tbl_desc = pd.read_excel(data_description_file, sheet_name='Sheet1', engine='openpyxl')
    tbl_summary = pd.read_csv(data_summary_file)

    tbl_merged = pd.merge(tbl_summary, tbl_desc)

    # num_digits_l = int(np.ceil(np.log10(tbl_merged['l'])).max())
    
    tbl_brief = tbl_merged[['dataset']].copy()
    tbl_brief['N'] = tbl_merged['N']
    tbl_brief['l'] = tbl_merged['l']
    tbl_brief['samples'] = tbl_merged['tot_samples']
    p_col = tbl_merged['p_samples']
    n_col = tbl_merged['tot_samples'] - p_col
    tbl_brief['p/n'] = (p_col / n_col).transform('{:.2f}'.format)
    # for idx in tbl_merged.index:
    #     N = int(tbl_merged.loc[idx, 'N'])
    #     l = int(tbl_merged.loc[idx, 'l'])
    #     tbl_brief.loc[idx, 'size N x l'] = f'{N} x {l:{num_digits_l}d}'

    #     p_samples = int(tbl_merged.loc[idx, 'p_samples'])
    #     tot_samples = int(tbl_merged.loc[idx, 'tot_samples'])
    #     n_samples = tot_samples - p_samples
    #     tbl_brief.loc[idx, 'samples (p/n)'] = f'{tot_samples} ({p_samples/n_samples:0.2f})'
        
    tbl_brief['hc_col'] = tbl_merged['hc_col']
    for idx in tbl_merged.index:
        card = int(tbl_merged.loc[idx,'cardinality'])
        card_rank = int(tbl_merged.loc[idx,'card_rank'])
        tbl_brief.loc[idx,'card(r)'] = f'{card} ({card_rank})'

    tbl_brief['imp'] = tbl_merged['perm_imp'].transform('{:.3f}'.format)
    for idx in tbl_merged.index:
        rank_all = int(tbl_merged.loc[idx, 'perm_imp_rank'])
        rank_cat = int(tbl_merged.loc[idx, 'perm_imp_rank_cat'])
        tbl_brief.loc[idx, 'rank(c)'] = f'{rank_all}({rank_cat})'
    
    tbl_brief['source_URL'] = tbl_merged['source_URL']
    
    filename = os.path.join(dir_name,'data_summary_brief.csv')
    tbl_brief.to_csv(filename, index=False)
    return tbl_brief



def remove_time_limit_exceeded(df):
    idx = df['measure'] == 'time_limit_exceeded'
    if idx.sum() > 0:
        print('remove time_limit_exceeded records')
        df = df[~idx].copy()
        df['performance'] = df['performance'].astype(float)
    return df

# Merge experiment results, baseline results
# Compute relative performance
def merge_exp_results(baseline='one-hot'):
    merged_fname = dir_name+"/exp2_results_details.csv"
    baseline_fname = dir_name+'/exp2_results_baseline.csv'
    
    
    # detail_file_name = f'{dir_name}/exp2_results_details.csv'
    # df_exp = pd.read_excel(detail_file, sheet_name='Sheet0', engine='openpyxl')

    # Concatenate all experiment results into a single table df_exp
    # df_exp = pd.concat([
    #     pd.read_csv(exp_result_dir+'/fast_algos/fold=5_seed=0/summary.csv'),
    #     pd.read_csv(exp_result_dir+'/NeuralNet_small_sets/fold=5_seed=0/summary.csv'),
    #     pd.read_csv(exp_result_dir+'/NeuralNet_H1BVisa/fold=5_seed=0/summary.csv'),
    #     pd.read_csv(exp_result_dir+'/SVM_small_sets/fold=5_seed=0/summary.csv'),
    #     pd.read_csv(exp_result_dir+'/kNN/fold=5_seed=0/summary.csv')
    # ])
    import glob
    df_lst = []
    for f in glob.glob(exp_result_dir+'/*/fold=5_seed=0/summary.csv'):
        print(f'  loading exp result from: {f}')
        df = pd.read_csv(f)
        df = remove_time_limit_exceeded(df)
        df_lst.append(df)
    df_exp = pd.concat(df_lst)
    df_exp.to_csv(merged_fname, index=False)
    print(f'save merged result to {merged_fname}')

    # Concate all baseline results into a single table df_base        
    # df_base = pd.read_csv('../results/exp2_2022_04_16_Jratio=1_fast_algos_sensible_default/fold=5_seed=0/summary.csv')
    # df_base = pd.concat([
    #     pd.read_csv(exp_result_dir+'/baseline/one-hot_fast_algos/fold=5_seed=0/summary.csv'),
    #     pd.read_csv(exp_result_dir+'/baseline/one-hot_NeuralNet/fold=5_seed=0/summary.csv'),
    #     pd.read_csv(exp_result_dir+'/baseline/one-hot_SVM/fold=5_seed=0/summary.csv'),
    #     pd.read_csv(exp_result_dir+'/baseline/TargetEncoder_tree_algos/fold=5_seed=0/summary.csv')
    #     ])
    df_lst = []
    for f in glob.glob(exp_result_dir+'/baseline/*/fold=5_seed=0/summary.csv'):
        print(f'  loading baseline from: {f}')
        df = pd.read_csv(f)
        df = remove_time_limit_exceeded(df)
        df_lst.append(df)
    df_base = pd.concat(df_lst)
    df_base.to_csv(baseline_fname, index=False)
    print(f'save merged baseline to {baseline_fname}')
    
    # For tree algorithms, create two baseline using one-hot and TargetEncoder respectively
    #    exp_result hc_cat_encoder  --> baseline (hc_cat_encoder)
    #               pca             --> one-hot
    #               mhe             --> one-hot
    #               mhe+te          --> TargetEncoder
    # Join experiment results with baseline and compute relative performance
    key_cols = ['ds_name','hc_name','algo_name', 'other_cat_encoder','fold','measure']
    df_exp_mhe_te = df_exp[df_exp['hc_cat_encoder'] == 'mhe+te']
    df_base_te = df_base.loc[df_base['hc_cat_encoder'] == 'TargetEncoder',
                             key_cols+['hc_cat_encoder', 'performance']].copy()
    df_merged_mhe_te = pd.merge(df_exp_mhe_te, df_base_te, how='left', on=key_cols)

    df_exp_other = df_exp[df_exp['hc_cat_encoder'] != 'mhe+te']
    df_base_other = df_base.loc[df_base['hc_cat_encoder'] != 'TargetEncoder',
                             key_cols+['hc_cat_encoder', 'performance']].copy()
    df_merged_other = pd.merge(df_exp_other, df_base_other, how='left', on=key_cols)
    
    df_merged = pd.concat([df_merged_other, df_merged_mhe_te])
    df_merged['relative_performance'] = df_merged['performance_x'] / df_merged['performance_y']
    df_merged.rename(columns={'performance_x':'performance',
                              'performance_y':'performance_baseline',
                              'hc_cat_encoder_x':'hc_cat_encoder',
                              'hc_cat_encoder_y':'hc_cat_encoder_baseline'}, inplace=True)
    df_merged = df_merged.replace({'algo_name': algo_short_name_dict})
    df_merged = df_merged.replace({'ds_name': ds_short_name_dict})
    
    df_merged.to_csv(detailed_exp_results_fname, index=False)
    print(f'save exp results with relative performance to {detailed_exp_results_fname}')
        
    return df_exp,df_base,df_merged





# 按 ds_name, algo_name, hc_cat_encoder, J 分组计算每组的平均表现
# 找到 MHG 表现最好的 J*
#    筛选出 J* 下的详细实验结果
#    计算 J* 下 MHG 与 PCA 的平均表现
# mhg 与 pca 分别找最好的 J*, 用各自的 J* 下的最优性能进行比较
def create_mesuare_summary(measure='F1'):
    df_detail = pd.read_csv(detailed_exp_results_fname)
    
    mean_fname = f'{dir_name}/{measure}_mean.csv'
    bestJ_mhe_fname = f'{dir_name}/{measure}_bestJ_4MHE.csv'
    detail_4mhe_bestJ_fname = f'{dir_name}/exp2_results_details_4MHE_bestJ_{measure}.csv'
    mean_4mhe_bestJ_fname = f'{dir_name}/{measure}_mean_4MHE_bestJ.csv'
    detail_bestJ_fname = f'{dir_name}/exp2_results_details_bestJ_{measure}.csv'
    mean_bestJ_fname = f'{dir_name}/{measure}_mean_bestJ.csv'

    
    # 1. 按 'ds_name','algo_name','hc_cat_encoder','J' 分组，计算每组 5 fold 的平均值
    df_mean = df_detail.query(f'measure == "{measure}"').groupby(
        ['ds_name','algo_name','hc_cat_encoder','J'] 
    ).agg({
        'performance': ['mean', 'std'],
        'relative_performance': ['mean', 'std'],
        'performance_baseline': ['mean', 'std'],
        'measure': 'count'
    }).reset_index()  # groupby 中用到的列从 index 变回列
    # Columns are now two level index
    # level 0: 'ds_name', ..., 'performance', 'performance' 
    # level 1:  ''      ,    ,  'mean',       'std'
    # we join the two levels using '_' to form a single name (and remove the trailing '_')
    df_mean.columns = df_mean.columns.map('_'.join).str.strip('_')
    df_mean.to_csv(mean_fname, index=False)
    
    # 2. 按 ds_name, algo_name 分组； 每组找到使 mhe 的 performance 最大的 J
    df_bestJ_MHE = df_mean.loc[
        df_mean.query('hc_cat_encoder == "mhe"').groupby( # a. 筛选出 mhe 的结果
            ['ds_name','algo_name'] # b. 按 ds_name, algo_name 分组
        )['performance_mean'].idxmax()   # c. 每组按照 performance 列找到最大值的行号
        , ['ds_name','algo_name','J']
    ]                               # d. 选出每组最大 performance 对应的行
    df_bestJ_MHE.to_csv(bestJ_mhe_fname, index=False)
    
    # 3. 每个 ds_name, algo_name, hc_encoder 只选出 mhe 的 bestJ 对应的结果 
    df_detail_4MHE_bestJ = pd.merge(df_bestJ_MHE, df_detail, how='left')
    df_detail_4MHE_bestJ.to_csv(detail_4mhe_bestJ_fname, index=False)
    df_mean_4MHE_bestJ = pd.merge(df_bestJ_MHE, df_mean, how='left')
    df_mean_4MHE_bestJ.to_csv(mean_4mhe_bestJ_fname, index=False)

    # 4. 按 ds_name, algo_name, hc_cat_encoder 分组； 每组找到 performance 最大的 J
    df_mean_bestJ = df_mean.loc[
        df_mean.groupby(
            ['ds_name','algo_name','hc_cat_encoder'] # a. 按 ds_name, algo_name, hc_cat_encoder 分组
        )['performance_mean'].idxmax()   # b. 每组按照 performance 列找到最大值的行号
    ]                               # c. 选出每组最大 performance 对应的行
    df_mean_bestJ.to_csv(mean_bestJ_fname, index=False)
    
    # 5. 按 ds_name, algo_name, hc_cat_encoder 分组； 每组找到 performance 最大的 J
    df_bestJ = df_mean_bestJ[['ds_name', 'algo_name', 'hc_cat_encoder', 'J']]
    df_detail_bestJ = pd.merge(df_bestJ, df_detail, how='left')
    df_detail_bestJ.to_csv(detail_bestJ_fname, index=False)
    

    



# 数据集分组，按顺序展示    
def plot_algos_per_dataset_ordered(hc_cat_encoder='mhe', measure='F1', performance_col='relative_performance_mean',
                                   ds_order = ['Crime', 'Colleges', 'Road', 'Visa', 'Employ', 'Loan'],
                                   algo_order = ['RF', 'GBDT', 'SVM', 'LR', 'MLP', 'DT'],
                                   save_dir = None):
    if save_dir is None:
        save_dir = os.path.join(dir_name, f"algos_per_dataset_plots_{measure}_{hc_cat_encoder}_{performance_col}")
    os.makedirs(save_dir, exist_ok=True)
    
    df_summary = pd.read_csv(data_summary_file, index_col=1)
    df_summary.replace({'ds_name': ds_short_name_dict}, inplace=True)
    
    df_mean = pd.read_csv(os.path.join(dir_name,measure+"_mean.csv"))
    df_mean = df_mean.query(f'hc_cat_encoder=="{hc_cat_encoder}"')
    df_mean = df_mean.loc[~df_mean[performance_col].isna()]

    df_mean = df_mean.loc[df_mean['ds_name'].isin(ds_order)]
    df_mean = df_mean.loc[df_mean['algo_name'].isin(algo_order)]

    # 设置默认绘图风格
    plt.style.use("ggplot") 
    
    
    min_y = df_mean[performance_col].min() * 100
    max_y = df_mean[performance_col].max() * 100
    range_y = max_y - min_y

    min_range_y = 10
    if range_y < min_range_y:
        mid_y = (min_y + max_y) / 2
        min_y,max_y = mid_y - min_range_y/2, mid_y + min_range_y/2
    else:
        min_y,max_y = min_y - range_y * 0.2, max_y + range_y * 0.2
    

    ds_groups = df_mean.groupby(['ds_name'])
    for ds_name in ds_order:
        try:
            data = ds_groups.get_group(ds_name)
            print(f'plot {ds_name}')
        except KeyError:
            continue
 
        imp = df_summary.loc[ds_name, 'perm_imp']
        # rank = int(df_summary.loc[ds_name, 'perm_imp_rank'])
        # cardinality = int(df_summary.loc[ds_name, 'cardinality'])
        
        samples_per_cat = df_summary.loc[ds_name, 'samples_per_cat']

        title = f'{ds_name} imp: {imp:.3f} #/cat: {samples_per_cat:.1f}'
        plt.figure(figsize=(6, 3), dpi=300)
        plt.title(title)
        plt.tick_params(labelsize=14)
        plt.grid(linestyle=":")  
        # plt.ylabel(f'{measure}', fontsize=18)
        # plt.xlabel('J', fontsize=18)
        plt.ylim(min_y, max_y)
        
        min_x = data['J'].min()
        max_x = data['J'].max()
        range_x = max_x - min_x
        
        plt.xlim(min_x - range_x * 0.05, max_x + range_x * 0.35)

        algo_groups = data.groupby('algo_name')
        for algo_name in algo_order:
            try:
                algo_data = algo_groups.get_group(algo_name, obj=None)
                algo_data = algo_data.sort_values(by='J')
                mhe = algo_data[performance_col] * 100
                plt.plot(algo_data['J'], mhe, linewidth=2, label=algo_name)
            except KeyError:
                continue
        plt.legend()
        
        file_name = os.path.join(save_dir, f"{ds_name}.png")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()    


    
        plt.figure(figsize=(4.8, 3), dpi=300)
        plt.title(title)
        plt.tick_params(labelsize=14)
        plt.grid(linestyle=":")  
        plt.ylim(min_y, max_y)
        plt.xlim(min_x - range_x * 0.05, max_x + range_x * 0.05)

        algo_groups = data.groupby('algo_name')
        for algo_name in algo_order:
            try:
                algo_data = algo_groups.get_group(algo_name, obj=None)
                algo_data = algo_data.sort_values(by='J')
                mhe = algo_data[performance_col] * 100
                plt.plot(algo_data['J'], mhe, linewidth=2, label=algo_name)
            except KeyError:
                continue
        file_name = os.path.join(save_dir, f"{ds_name}_no_legend.png")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()    



# 算法分组，按顺序展示    
def plot_datasets_per_algo_ordered(hc_cat_encoder='mhe', measure='F1', performance_col='relative_performance_mean',
                                   ds_order = ['Adult', 'Crime', 'Colleges', 'Road', 'Visa', 'Employ', 'Loan', 'Traffic'],
                                   algo_order = ['RF', 'GBDT', 'SVM', 'LR', 'MLP', 'DT'],
                                   save_dir = None):
    if save_dir is None:
        save_dir = os.path.join(dir_name, f"datasets_per_algo_plots_{measure}_{hc_cat_encoder}_{performance_col}")
    os.makedirs(save_dir, exist_ok=True)
    
    df_summary = pd.read_csv(data_summary_file, index_col=1)
    df_summary.replace({'ds_name': ds_short_name_dict}, inplace=True)
    # df_summary.sort_values(by='cardinality', inplace=True) # sort dataset in ascending order of cardinality 
    
    df_mean = pd.read_csv(os.path.join(dir_name,measure+"_mean.csv"))
    df_mean = df_mean.query(f'hc_cat_encoder=="{hc_cat_encoder}"')
    df_mean = df_mean.loc[~df_mean[performance_col].isna()]

    df_mean = df_mean.loc[df_mean['ds_name'].isin(ds_order)]
    df_mean = df_mean.loc[df_mean['algo_name'].isin(algo_order)]

    df_J = df_summary[['ds_name', 'cardinality']]    
    df_mean = pd.merge(df_mean, df_J, how='left', on='ds_name')
    df_mean['J_ratio'] = df_mean['J'] / df_mean['cardinality']

    # 设置默认绘图风格
    plt.style.use("ggplot") 
    
    
    min_y = df_mean[performance_col].min() * 100
    max_y = df_mean[performance_col].max() * 100
    range_y = max_y - min_y

    min_range_y = 10
    if range_y < min_range_y:
        mid_y = (min_y + max_y) / 2
        min_y,max_y = mid_y - min_range_y/2, mid_y + min_range_y/2
    else:
        min_y,max_y = min_y - range_y * 0.2, max_y + range_y * 0.2
    

    algo_groups = df_mean.groupby(['algo_name'])
    for algo_name in algo_order:
        try:
            algo_data = algo_groups.get_group(algo_name)
            print(f'plot {algo_name}')
        except KeyError:
            continue
 
        title = f'{algo_name} relative {measure}'
        plt.figure(figsize=(6, 3), dpi=300)
        plt.title(title)
        plt.tick_params(labelsize=14)
        plt.grid(linestyle=":")  
        # plt.ylabel(f'{measure}', fontsize=18)
        # plt.xlabel('J', fontsize=18)
        plt.ylim(min_y, max_y)
        
        min_x = algo_data['J_ratio'].min()
        max_x = algo_data['J_ratio'].max()
        range_x = max_x - min_x
        
        plt.xlim(min_x - range_x * 0.05, max_x + range_x * 0.35)

        ds_groups = algo_data.groupby('ds_name')
        for ds_name in ds_order:
            try:
                ds_data = ds_groups.get_group(ds_name)
                ds_data = ds_data.sort_values(by='J')
                mhe = ds_data[performance_col] * 100
                plt.plot(ds_data['J_ratio'], mhe, linewidth=2, label=ds_name)
            except KeyError:
                continue
        plt.legend()
        
        file_name = os.path.join(save_dir, f"{algo_name}.png")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()    





# ds_vs_algo@MHE_bestJ_avg_relative_{measure}.csv:
#   each cell is the average relative perofrmance (std)
# ds_vs_algo@MHE_bestJ_avg_{measure}.csv:
#   each cell is the average performance (average baseline performance)
# ds_vs_algo@MHE_bestJ_avg_relative_training_time.csv:
#   each cell is the bestJ (average relative training time) 
# ds_vs_algo@MHE_bestJ_alpha_avg_relative_training_time.csv
#   each cell is the J/I (average relative training time)
def create_ds_vs_algo_bestJ_4MHE(measure='F1', hc_cat_encoder='mhe',
                ds_order = ['Adult', 'Crime', 'Colleges', 'Road', 'Visa', 'Employ', 'Loan', 'Traffic'],
                algo_order = ['RF', 'GBDT', 'SVM', 'LR', 'MLP', 'DT']):
    F1_ds_vs_algo_fname = f'{dir_name}/ds_vs_algo@MHE_bestJ_avg_relative_{measure}.csv'
    F1_abs_ds_vs_algo_fname = f'{dir_name}/ds_vs_algo@MHE_bestJ_avg_{measure}.csv'
    tt_ds_vs_algo_fname = f'{dir_name}/ds_vs_algo@MHE_bestJ_avg_relative_training_time.csv'
    alpha_tt_ds_vs_algo_fname = f'{dir_name}/ds_vs_algo@MHE_bestJ_alpha_avg_relative_training_time.csv'

    
    detail_4mhe_bestJ_fname = f'{dir_name}/exp2_results_details_4MHE_bestJ_{measure}.csv'
    df_detail_4MHE_bestJ = pd.read_csv(detail_4mhe_bestJ_fname)
    df_detail_4MHE_bestJ = df_detail_4MHE_bestJ.query(f'hc_cat_encoder == "{hc_cat_encoder}"')

    df_F1 = df_detail_4MHE_bestJ.query(f'measure == "{measure}"').groupby(
        ['ds_name','algo_name'] 
    ).agg({
        'relative_performance': ['mean', 'std'],
        'performance': ['mean', 'std'],
        'performance_baseline': ['mean', 'std'],
        'measure': 'count'
    }).reset_index()  # groupby 中用到的列从 index 变回列
    # ('relative_performance', 'mean') --> 'relative_performance_mean'
    df_F1.columns = df_F1.columns.map('_'.join).str.strip('_')

    df_F1_ds_vs_algo = pd.DataFrame()  # average relative performance
    df_F1_abs_ds_vs_algo = pd.DataFrame() # average absolute performance
    groups = df_F1.groupby(['ds_name', 'algo_name'])
    ds_to_better = {}
    ds_to_worse = {}
    algo_to_better = {}
    algo_to_worse = {}
    for ds_name in ds_order:
        for algo_name in algo_order:
            try:
                data = groups.get_group((ds_name, algo_name))
                assert len(data) == 1
            except KeyError:
                continue
            
            relative_performance_mean = data['relative_performance_mean'].iloc[0]
            relative_performance_std = data['relative_performance_std'].iloc[0]
            cell = f'{relative_performance_mean:.4f} ({relative_performance_std:.4f})'
            if relative_performance_mean > 1 + relative_performance_std:
                cell = cell + '+'
                ds_to_better[ds_name] = ds_to_better.get(ds_name, 0)+1
                algo_to_better[algo_name] = algo_to_better.get(algo_name, 0)+1
            elif relative_performance_mean < 1 - relative_performance_std:
                cell = cell + '-'
                ds_to_worse[ds_name] = ds_to_worse.get(ds_name, 0)+1
                algo_to_worse[algo_name] = algo_to_worse.get(algo_name, 0)+1
            # print(f'{ds_name}, {algo_name}, {len(data)=}, {cell}')

            df_F1_ds_vs_algo.loc[ds_name, algo_name] = cell


            performance_mean = data['performance_mean'].iloc[0]
            performance_baseline_mean = data['performance_baseline_mean'].iloc[0]
            cell = f'{performance_mean:.4f} ({performance_baseline_mean:.4f})'
            df_F1_abs_ds_vs_algo.loc[ds_name, algo_name] = cell
            
    for ds_name in ds_order:
        df_F1_ds_vs_algo.loc[ds_name, '#o'] = ds_to_better.get(ds_name, 0)
        df_F1_ds_vs_algo.loc[ds_name, '#u'] = ds_to_worse.get(ds_name, 0)
    for algo_name in algo_order:
        df_F1_ds_vs_algo.loc['#o', algo_name] = algo_to_better.get(algo_name, 0)
        df_F1_ds_vs_algo.loc['#u', algo_name] = algo_to_worse.get(algo_name, 0)
    
    df_F1_ds_vs_algo.reset_index(inplace=True)
    df_F1_ds_vs_algo.rename(columns={'index':'dataset'}, inplace=True)
    df_F1_ds_vs_algo.to_csv(F1_ds_vs_algo_fname, index=False)

    df_F1_abs_ds_vs_algo.reset_index(inplace=True)
    df_F1_abs_ds_vs_algo.rename(columns={'index':'dataset'}, inplace=True)
    df_F1_abs_ds_vs_algo.to_csv(F1_abs_ds_vs_algo_fname, index=False)




    df_tt = df_detail_4MHE_bestJ.query('measure == "train_time(s)"').groupby(
        ['ds_name','algo_name'] 
    ).agg({
        'relative_performance': ['mean', 'std'],
        'measure': 'count',
        'J': 'first',
    }).reset_index()  # groupby 中用到的列从 index 变回列
    # ('relative_performance', 'mean') --> 'relative_performance_mean'
    df_tt.columns = df_tt.columns.map('_'.join).str.strip('_')
    
    df_summary = pd.read_csv(data_summary_file, index_col=1)
    df_summary.replace({'ds_name': ds_short_name_dict}, inplace=True)
    df_J = df_summary[['ds_name', 'cardinality']]    
    df_tt = pd.merge(df_tt, df_J, how='left', on='ds_name')
    df_tt['J_ratio'] = df_tt['J_first'] / df_tt['cardinality']

    

    df_tt_ds_vs_algo = pd.DataFrame()    
    df_alpha_tt_ds_vs_algo = pd.DataFrame()    
    groups = df_tt.groupby(['ds_name', 'algo_name'])
    for ds_name in ds_order:
        for algo_name in algo_order:
            try:
                data = groups.get_group((ds_name, algo_name))
                assert len(data) == 1
            except KeyError:
                continue

            J = int(data['J_first'].iloc[0])
            relative_performance_mean = data['relative_performance_mean'].iloc[0]
            cell = f'{J:4d} ({relative_performance_mean:.4f})'
            df_tt_ds_vs_algo.loc[ds_name, algo_name] = cell
            
            J_ratio = data['J_ratio'].iloc[0]
            cell = f'{J_ratio:.4f} ({relative_performance_mean:.4f})'
            df_alpha_tt_ds_vs_algo.loc[ds_name, algo_name] = cell
    
    df_tt_ds_vs_algo.reset_index(inplace=True)
    df_tt_ds_vs_algo.rename(columns={'index':'dataset'}, inplace=True)
    df_tt_ds_vs_algo.to_csv(tt_ds_vs_algo_fname, index=False)

    df_alpha_tt_ds_vs_algo.reset_index(inplace=True)
    df_alpha_tt_ds_vs_algo.rename(columns={'index':'dataset'}, inplace=True)
    df_alpha_tt_ds_vs_algo.to_csv(alpha_tt_ds_vs_algo_fname, index=False)














def plot_mhe_vs_pca_on_dataset_algo(
        measure='F1', sub_plot=True,
        performance_col='performance_mean',
        ds_order = ['Adult', 'Crime', 'Colleges', 'Road', 'Visa', 'Employ', 'Loan', 'Traffic'],
        algo_order = ['RF', 'GBDT', 'SVM', 'LR', 'MLP', 'DT']):
    tbl_summary = pd.read_csv(data_summary_file, index_col=1)
    
    tbl_mean = pd.read_csv(os.path.join(dir_name,measure+"_mean.csv"))
    tbl_mean = tbl_mean.loc[~tbl_mean[performance_col].isna()]
    # tbl_mean = tbl_mean.replace({'algo_name': algo_short_name_dict})
    # tbl_mean = tbl_mean.replace({'ds_name': ds_short_name_dict})

    # 设置默认绘图风格
    plt.style.use("ggplot") 

    ds_name_to_idx = {}
    ds_count = len(ds_order)
    for idx, ds_name in enumerate(ds_order): # tbl_mean['ds_name'].unique():
        ds_name_to_idx[ds_name] = idx
    algo_name_to_idx = {}
    algo_count = len(algo_order)
    for idx, algo_name in enumerate(algo_order): # tbl_mean['algo_name'].unique():
        algo_name_to_idx[algo_name] = idx

    if sub_plot:    
        plt.figure(figsize=(6*algo_count, 3*ds_count), dpi=300)
    else:
        plt.figure(figsize=(6, 3), dpi=300)

    groups = tbl_mean.groupby(['ds_name','algo_name'])
    for ds_name in ds_order:
        for algo_name in algo_order:
            try:
                data = groups.get_group((ds_name,algo_name))
                print(f'plot {ds_name} {algo_name}')
            except KeyError:
                continue
            
            min_y = data[performance_col].min() * 100
            max_y = data[performance_col].max() * 100
            range_y = max_y - min_y
    
            min_x = data['J'].min()
            max_x = data['J'].max()
            range_x = max_x - min_x
            
            pivot = data.pivot(index='J',columns='hc_cat_encoder',values=[performance_col])
            pca = pivot[(performance_col,'pca')].apply(lambda x: 100*x)
            mhe = pivot[(performance_col,'mhe')].apply(lambda x: 100*x)
            
            ds_idx = ds_name_to_idx[ds_name]
            algo_idx = algo_name_to_idx[algo_name]
            sub_plot_idx = algo_idx * ds_count + ds_idx + 1
            if sub_plot:
                plt.subplot(algo_count, ds_count, sub_plot_idx)
                plt.subplots_adjust(hspace=0.01)
             
            # plt.title(title, fontsize = 14)
            plt.tick_params(labelsize=14)
            plt.grid(linestyle=":")  
            if not sub_plot or ds_idx == 0:
                plt.ylabel(f'{algo_name} {measure}', fontsize=18)
            if not sub_plot or algo_idx == algo_count - 1:
                imp = tbl_summary.loc[ds_name, 'perm_imp']
                rank = int(tbl_summary.loc[ds_name, 'perm_imp_rank'])
                plt.xlabel(f"{ds_name}: imp {imp:.2f} ({rank})", fontsize=18)
            else:
                plt.xticks([])          
            
            J_count = len(mhe)
            plt.plot(pivot.index, mhe, c='r', linewidth=2, label='mhe')
            plt.plot(pivot.index, pca, c='b', linewidth=2, label='pca')
            max_v = max(mhe.max(), pca.max())
            plt.plot(pivot.index, [max_v]*J_count, c='black', linewidth=2, label='best')
            if performance_col.startswith('relative'):
                base = 1.0
            else:
                base = data['performance_baseline_mean'].iloc[0]
            plt.plot(pivot.index, [base*100]*J_count, c='g', linewidth=2, label='base')
            plt.legend()
    
            # plt.ylim(min_y - range_y * 0.2, max_y + range_y * 0.2)
            if range_y < 20:
                mid_y = (min_y + max_y) / 2
                plt.ylim(mid_y - 10, mid_y + 10)
            else:
                plt.ylim(min_y - range_y * 0.2, max_y + range_y * 0.2)
            plt.xlim(min_x - range_x * 0.2, max_x + range_x * 0.2)
            
            if not sub_plot:
                save_dir = os.path.join(dir_name, f'{measure}_{performance_col}_J_plots')
                os.makedirs(save_dir, exist_ok=True)
                file_name = os.path.join(save_dir, f"{ds_name}_{algo_name}.png")
                plt.savefig(file_name, bbox_inches='tight')
                plt.show()    

    if sub_plot:
        file_name = os.path.join(dir_name, measure+'_'+performance_col+"_grid.png")
        plt.savefig(file_name, bbox_inches='tight')
        plt.show()    
                   


# mhe_vs_pca@bestJ_avg_{measure}.csv:
#   each cell: avg. mhe performance - avg. pca performance  (std diff)
def create_mhe_vs_pca_at_bestJ(measure='F1', 
                ds_order = ['Adult', 'Crime', 'Colleges', 'Road', 'Visa', 'Employ', 'Loan', 'Traffic'],
                algo_order = ['RF', 'GBDT', 'SVM', 'LR', 'MLP', 'DT']):
    F1_mhe_vs_pca_ds_by_algo_fname = f'{dir_name}/mhe_vs_pca_ds_by_algo@bestJ_avg_diff_{measure}.csv'

    df_detail_bestJ = pd.read_csv(f'{dir_name}/exp2_results_details_bestJ_{measure}.csv')

    hc_cat_encoder=['mhe','pca']
    df_F1 = df_detail_bestJ.loc[(df_detail_bestJ['measure'] == measure) &
        df_detail_bestJ['hc_cat_encoder'].isin(hc_cat_encoder) &
        df_detail_bestJ['ds_name'].isin(ds_order) &
        df_detail_bestJ['algo_name'].isin(algo_order)]

    df_F1_diff = df_F1.pivot(index=['ds_name','algo_name','fold'], columns='hc_cat_encoder',
                        values=['performance']).reset_index()
    df_F1_diff.columns = df_F1_diff.columns.map('_'.join).str.strip('_')
    df_F1_diff['performance_diff'] = df_F1_diff['performance_mhe'] - df_F1_diff['performance_pca']
    df_F1_diff['performance_diff_%'] = df_F1_diff['performance_diff'] / df_F1_diff['performance_pca']
    

    df_diff_mean = df_F1_diff.groupby(
        ['ds_name','algo_name'] 
    ).agg({
        'performance_diff': ['mean', 'std', 'count'],
        'performance_diff_%': ['mean'],
        'performance_mhe': ['mean', 'std'],
        'performance_pca': ['mean', 'std']
    }).reset_index()  # groupby 中用到的列从 index 变回列
    # ('relative_performance', 'mean') --> 'relative_performance_mean'
    df_diff_mean.columns = df_diff_mean.columns.map('_'.join).str.strip('_')

    df_F1_mhe_vs_pca_ds_by_algo = pd.DataFrame()  
    groups = df_diff_mean.groupby(['ds_name', 'algo_name'])
    ds_to_better = {}
    ds_to_worse = {}
    algo_to_better = {}
    algo_to_worse = {}
    for ds_name in ds_order:
        for algo_name in algo_order:
            try:
                data = groups.get_group((ds_name, algo_name))
                assert len(data) == 1
            except KeyError:
                continue
            
            performance_diff_percent_mean = data['performance_diff_%_mean'].iloc[0]
            performance_diff_mean = data['performance_diff_mean'].iloc[0]
            performance_diff_std = data['performance_diff_std'].iloc[0]
            cell = f'{performance_diff_percent_mean:+.2%}'
            if performance_diff_mean > 0 + performance_diff_std:
                cell = cell + '*'
                ds_to_better[ds_name] = ds_to_better.get(ds_name, 0)+1
                algo_to_better[algo_name] = algo_to_better.get(algo_name, 0)+1
            elif performance_diff_mean < 0 - performance_diff_std:
                cell = cell + '*'
                ds_to_worse[ds_name] = ds_to_worse.get(ds_name, 0)+1
                algo_to_worse[algo_name] = algo_to_worse.get(algo_name, 0)+1
            else:
                cell = cell + '='
            # print(f'{ds_name}, {algo_name}, {len(data)=}, {cell}')

            df_F1_mhe_vs_pca_ds_by_algo.loc[ds_name, algo_name] = cell

            
    for ds_name in ds_order:
        df_F1_mhe_vs_pca_ds_by_algo.loc[ds_name, '#o'] = ds_to_better.get(ds_name, 0)
        df_F1_mhe_vs_pca_ds_by_algo.loc[ds_name, '#u'] = ds_to_worse.get(ds_name, 0)
    for algo_name in algo_order:
        df_F1_mhe_vs_pca_ds_by_algo.loc['#o', algo_name] = algo_to_better.get(algo_name, 0)
        df_F1_mhe_vs_pca_ds_by_algo.loc['#u', algo_name] = algo_to_worse.get(algo_name, 0)
    
    df_F1_mhe_vs_pca_ds_by_algo.reset_index(inplace=True)
    df_F1_mhe_vs_pca_ds_by_algo.rename(columns={'index':'dataset'}, inplace=True)
    df_F1_mhe_vs_pca_ds_by_algo.to_csv(F1_mhe_vs_pca_ds_by_algo_fname, index=False)










if __name__ == '__main__': 

    # create data_summary.csv and data_summary_brief.csv
    tbl_summary = create_data_summary() # about 5 minutes
    tbl_brief = create_data_summary_brief() # first Table in Section 4

    # merge experiment results, compute relative performance with baseline = 'one-hot' for hc_cat_encoder
    merge_exp_results()

    # prepare data for plotting    
    create_mesuare_summary(measure='F1')    
    
    # MHG+one-hot: Each dataset, one line per algo, avg F1 vs J
    plot_algos_per_dataset_ordered(ds_order=['Adult'])    
    plot_algos_per_dataset_ordered()    
    plot_algos_per_dataset_ordered(ds_order=['Traffic'])    

    # MHG+one-hot: For typical dataset Traffic, plot relative training time(s)    
    create_mesuare_summary(measure='train_time(s)')    
    plot_algos_per_dataset_ordered(measure='train_time(s)', 
                                    ds_order=['Traffic'],
                                    algo_order=['RF', 'GBDT', 'DT'],
                                    save_dir = dir_name+'/algos_per_dataset_plots_mhe_training_time_tree_algos')    
    plot_algos_per_dataset_ordered(measure='train_time(s)', 
                                    ds_order=['Traffic'],
                                    algo_order=['SVM', 'LR'],
                                    save_dir = dir_name+'/algos_per_dataset_plots_mhe_training_time_SVM_LR')    
    plot_algos_per_dataset_ordered(measure='train_time(s)', ds_order = ['Adult', 'Crime', 'Colleges', 'Road', 'Visa', 'Employ', 'Loan', 'Traffic'])

    # # MHG+one-hot: For each algo, plot its relative F1 peformance and relative training time on various dataset    
    # # Not uesful
    # plot_datasets_per_algo_ordered(measure='F1')
    # plot_datasets_per_algo_ordered(measure='train_time(s)')
    
    
    # MHG+one-hot: For each pair of dataset, classifier compute avg relative F1 under best J
    create_ds_vs_algo_bestJ_4MHE()
    

    # # one-hot+PCA: Each dataset, one line per algo, avg F1 vs J
    # # Not useful
    # plot_algos_per_dataset_ordered(hc_cat_encoder='pca', ds_order=['Adult'])    
    # plot_algos_per_dataset_ordered(hc_cat_encoder='pca')    
    # plot_algos_per_dataset_ordered(hc_cat_encoder='pca', ds_order=['Traffic'])    


    # MHG+one-hot vs one-hot+PCA table: row: dataset; col: algo
    create_mhe_vs_pca_at_bestJ()


    # MHG+one-hot vs one-hot+PCA, row: algo; col: dataset; x-cord: J; y-cord: avg relative F1
    plot_mhe_vs_pca_on_dataset_algo(measure='F1') # performance_col='relative_performance_mean' is less informative
    plot_mhe_vs_pca_on_dataset_algo(measure='F1', sub_plot=False) 
    

    
    


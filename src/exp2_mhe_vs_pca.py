# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:06:52 2021

@author: YingFu

对每个数据集的 hc 列，
1. 用 MaxHomoEncoder 维降到 J 组后用 one-hot 编码训练
2. 用 one-hot 编码后用 pca 降到 J 维

尝试以下数据集:
    1) TODO
    
尝试以下分类器的默认参数：
    1) Logistic_Regression
    2) Decision_Tree_Classifier
    3) Nearest_Neighbors
    4) Gaussian_Naive_Bayes
    5) Random_Forest
    6) GBDT
    7) MLP
    8) SVC

每个数据集用 7：3 随机划分 5 次 训练集 与 测试集
"""

import pandas as pd
import time
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
from sklearn import metrics

import datasets

# 实验用的模型列表： [(模型名字、模型、other_cat_encoder), ..., ]
# 这个 py 文件比较的是 sklearn 中各个 classifier 的默认参数

def models_sensible_default(seed=1):
    return [
        ('NeuralNet', MLPClassifier(
            # hidden_layer_sizes = (100,), activation = 'relu',
            # solver = "adam", # stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
            # alpha = 0.0001, # L2 penalty for regularization term
            # batch_size = 'auto', # minibatch size = min(200, n_samples)                                    
            learning_rate = 'adaptive', # keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.
            # learning_rate_init = 0.001
            max_iter=1000, # Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.
            # shuffle = True, # Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.
            # tol = 1e-4, # Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.
            # early_stopping = False, #  If early stopping is False, then the training stops when the training loss does not improve by more than tol for n_iter_no_change consecutive passes over the training set. Only effective when solver=’sgd’ or ‘adam’.
            # beta_1 = 0.9, # Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver=’adam’.
            # beta_2 = 0.999, # Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver=’adam’.
            # epsilon = 1e-8, # Value for numerical stability in adam. Only used when solver=’adam’.
            # n_iter_no_change = 10, # Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’.
            random_state=seed), 'one-hot'),
        ('GBDT', GradientBoostingClassifier(
            # loss = deviance, # deviance (= logistic regression) for classification with probabilistic outputs
            # learning_rate = 0.1, # Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.                             
            # n_estimators = 100, # The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
            # subsample = 1.0, # The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
            # criterion = 'friedman_mse', # The function to measure the quality of a split. 'Friedman_mse', is the selection criterion on page 12, equation (35) (http://luthuli.cs.uiuc.edu/~daf/courses/Opt-2017/Papers/2699986.pdf) which is used by LogitBoost. According to Friedman, using unit weight i.e., ordinary MSE is numerically more stable than. 
            # min_samples_split = 2, # The minimum number of samples required to split an internal node
            # min_samples_leaf = 1, # The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
            # min_weight_fraction_leaf = 0.0, # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
            # max_depth = 3, # The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
            # min_impurity_decrease = 0.0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            # max_features = None, # The number of features to consider when looking for the best split: If None, then max_features=n_features.
            # max_leaf_nodes = None, # Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            # validation_fraction = 0.1, # The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.
            # n_iter_no_change = None, # n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations. The split is stratified.
            # tol = 1e-4, # Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.
            # ccp_alpha = 0, # Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.
            random_state = seed # Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split (see Notes for more details). It also controls the random splitting of the training data to obtain a validation set if n_iter_no_change is not None. Pass an int for reproducible output across multiple function calls. See Glossary.
            ),'TargetEncoder'),               
        ('RBFSVM', SVC(
            # C = 1, # Regularization parameter. 
            # kernel = ’rbf’, # Specifies the kernel type to be used in the algorithm
            # gamma = 'scale', # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma
            # shrinking = True, # Whether to use the shrinking heuristic
            # probability = None, # Whether to enable probability estimates
            # tol = 1e-3, # Tolerance for stopping criterion
            # cache_size = 200, # Specify the size of the kernel cache (in MB).
            # class_weight = None, 
            # max_iter = -1, # Hard limit on iterations within solver, or -1 for no limit
            random_state = seed # Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when probability is False
            ), 'one-hot'),         
        ('DT', DecisionTreeClassifier(
            # criterion = 'gini', # The function to measure the quality of a split
            # splitter = 'best', # The strategy used to choose the split at each node
            max_depth=30, # The maximum depth of the tree
            min_samples_split = 20, # The minimum number of samples required to split an internal node
            # min_samples_leaf = 1, # The minimum number of samples required to be at a leaf node
            # max_leaf_nodes = None, # unlimited number of leaf nodes.
            # min_impurity_decrease = 0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            # class_weight = None, # Weights associated with classes in the form {class_label: weight}
            # ccp_alpha = 0, # no pruning
            random_state = seed # Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to "best" (affect tie breaking).
            ),'TargetEncoder'), 
        ('kNN', KNeighborsClassifier(
            # neighbors = 5, # Number of neighbors to use by default for kneighbors queries
            # weights = 'uniform', # All points in each neighborhood are weighted equally.
            # algorithm = 'auto', # attempt to decide the most appropriate algorithm based on the values passed to fit method.
            # leaf_size = 30, # Leaf size passed to BallTree or KDTree
            # p = 2, # Power parameter for the Minkowski metric, euclidean distance
            # metric = 'minkowski', # The distance metric to use for the tree
            ), 'one-hot'),        
        ('RF', RandomForestClassifier(
            # n_estimators=100, # The number of trees.
            # criterion = 'gini', # The function to measure the quality of a split
            max_depth=30, # The maximum depth of the tree
            min_samples_split = 20, # The minimum number of samples required to split an internal node
            # min_samples_leaf = 1, # The minimum number of samples required to be at a leaf node
            # max_features = 'auto', # The number of features to consider when looking for the best split = sqrt(n_features)
            # max_leaf_nodes = None, # unlimited number of leaf nodes.
            # min_impurity_decrease = 0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            # class_weight = None, # Weights associated with classes in the form {class_label: weight}
            # ccp_alpha = 0, # no pruning
            random_state = seed # Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to "best" (affect tie breaking).
            ),'TargetEncoder'),         
        ('LR', LogisticRegression(
            # penalty = 'l2', # add a L2 penalty term
            # dual = False, # Dual or primal formulation
            # tol = 1e-4, # Tolerance for stopping criteria
            # C = 1.0, # Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
            # fit_intercept = True, # Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
            # intercept_scaling = 1.0, # Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling]
            # class_weight = None, # all classes are supposed to have weight one.
            random_state = seed, # Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data
            # solver = 'lbfgs', 
            # max_iter = 100, # Maximum number of iterations taken for the solvers to converge
            ), 'one-hot'), 
        ('GBDT_MSE', GradientBoostingClassifier(
            # loss = deviance, # deviance (= logistic regression) for classification with probabilistic outputs
            # learning_rate = 0.1, # Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.                             
            # n_estimators = 100, # The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
            # subsample = 1.0, # The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
            criterion = 'mse', # The function to measure the quality of a split. 'Friedman_mse', is the selection criterion on page 12, equation (35) (http://luthuli.cs.uiuc.edu/~daf/courses/Opt-2017/Papers/2699986.pdf) which is used by LogitBoost. According to Friedman, using unit weight i.e., ordinary MSE is numerically more stable than. 
            # min_samples_split = 2, # The minimum number of samples required to split an internal node
            # min_samples_leaf = 1, # The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
            # min_weight_fraction_leaf = 0.0, # The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
            # max_depth = 3, # The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
            # min_impurity_decrease = 0.0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            # max_features = None, # The number of features to consider when looking for the best split: If None, then max_features=n_features.
            # max_leaf_nodes = None, # Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            # validation_fraction = 0.1, # The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.
            # n_iter_no_change = None, # n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving. By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations. The split is stratified.
            # tol = 1e-4, # Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.
            # ccp_alpha = 0, # Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.
            random_state = seed # Controls the random seed given to each Tree estimator at each boosting iteration. In addition, it controls the random permutation of the features at each split (see Notes for more details). It also controls the random splitting of the training data to obtain a validation set if n_iter_no_change is not None. Pass an int for reproducible output across multiple function calls. See Glossary.
            ),'TargetEncoder'),               
        ('RF_Fast', RandomForestClassifier(
            # n_estimators=100, # The number of trees.
            # criterion = 'gini', # The function to measure the quality of a split
            max_depth=30, # The maximum depth of the tree
            min_samples_split = 20, # The minimum number of samples required to split an internal node
            # min_samples_leaf = 1, # The minimum number of samples required to be at a leaf node
            # max_features = 'auto', # The number of features to consider when looking for the best split = sqrt(n_features)
            max_leaf_nodes = 50, # unlimited number of leaf nodes.
            bootstrap = True, # Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
            # min_impurity_decrease = 0, # A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
            # class_weight = None, # Weights associated with classes in the form {class_label: weight}
            # ccp_alpha = 0, # no pruning
            max_samples = 5000, # If bootstrap is True, the number of samples to draw from X to train each base estimator.
            random_state = seed # Controls the randomness of the estimator. The features are always randomly permuted at each split, even if splitter is set to "best" (affect tie breaking).
            ),'TargetEncoder'),
        ('BNB', BernoulliNB(), 'one-hot'), # not approperiate
        ('GNB', GaussianNB(), 'one-hot'), # not approperiate
        ('MNB', MultinomialNB(), 'one-hot'), # not approperiate
        ('LinearSVM', SVC(kernel = "linear"), 'one-hot') # poor performance
        ]

model_lst = models_sensible_default()

def get_model_by_name(names, models=model_lst):
    return [x for x in models if x[0] in names]    


def create_J_list(card, min_J, ratio_lst):
    '''
    Create a list of J, by int(card * ratio), remove duplicate and keep J >= min_J
    '''
    used = set()
    J_lst = []
    for J in [int(card*ratio) for ratio in ratio_lst]:
        if J >= min_J and (J not in used):
            J_lst.append(J)
            used.add(J)
    return J_lst


# 有比较分散的 J, 相对密集的 J, 密集的 J
# 用 list 替代 dict 好处： 可以一个数据集，用两个 hc_col 来创建两个实验场景
def create_dataset_J_lst(min_J=2, ratio_lst = [0.1,0.08,0.06,0.04,0.5,0.4,0.3,0.2,0.02,0.016,0.012,0.008,0.004]):
    return [('Adult', 'native-country', create_J_list(42, min_J=min_J, ratio_lst=ratio_lst)),
            # ('Colleges', 'State', create_J_list(59, min_J=min_J, ratio_lst=ratio_lst)), 
            ('Colleges', 'Carnegie Basic Classification', create_J_list(34, min_J=min_J, ratio_lst=ratio_lst)), 
            ('EmployeeSalaries', 'Employee Position Title',create_J_list(385, min_J=min_J, ratio_lst=ratio_lst)),
            ('Crime', 'Crm Cd Desc', create_J_list(129, min_J=min_J, ratio_lst=ratio_lst)),
            ('TrafficViolations', 'Charge',create_J_list(525, min_J=min_J, ratio_lst=ratio_lst)),                 # take sampling
            ('H1BVisa', 'SOC_NAME' , create_J_list(763, min_J=min_J, ratio_lst=ratio_lst)),                       # take sampling
            ('AutoLoan', 'employee_code_id', create_J_list(3147, min_J=min_J, ratio_lst=ratio_lst)),              # take sampling
            ('RoadSafety', 'Local_Authority_(District)', create_J_list(380, min_J=min_J, ratio_lst=ratio_lst))    # take sampling
            # ('LendingClub', 'zip_code', [50, 40, 30, 25, 20, 15, 10]),       
            # ('OpenPayments', 'Recipient_City', [50, 45, 40, 35, 30, 25, 20, 15, 10]),
            ]
 
data_hc_Jlst = create_dataset_J_lst()

def get_data_Jlst_by_name(names, data_Jlst = data_hc_Jlst):
    return [x for x in data_Jlst if x[0] in names]    
    


def loader(data_hc_Jlst):
    for (ds_name,hc_col,JLst) in data_hc_Jlst:
        ds = datasets.load_one_set(ds_name)
        yield (ds,hc_col,JLst)    

from preprocessor import prepare_train_test
from sklearn.model_selection import ShuffleSplit
from timer import run_with_time_limit
def exp(models, foldsCount, data_hc_Jlst, loader=loader, 
        hc_cat_encoder_list = ['pca', 'mhe'], seed=0, result_dir="../results",
        time_limit=10800):
    '''
    对models中每个算法、data_hc_Jlst中每个数据集、每个J、对hc列的两种编码方式（pca、mhe)
        重复foldsCount次训练、测试
        
    训练及测试集中 hc 列的两种编码方式为
    1. hc_cat_encoder = pca ---对照组，先填充缺失、one-hot、pca降维到J
    2. hc_cat_encoder = mhe ---实验组，先填充缺失、MaxHomo变成J组、one-hot
    
                    
    models: list(tuple)
        每个模型由一个tuple记录
            [0]: 模型的名称
            [1]: 模型clf
            [2]: 数据集中hc之外的cat列的编码方式 other_cat_encoder （树分类算法应
                 该用 'TargetEncoder'，其他分类算法应该用 'one-hot'
    foldsCount: int
        每个设定下重复训练与测试次数
    data_hc_Jlst : list(tuple)
        每个数据集由一个tuple记录
            [0]: 数据集名字 （datasets 中定义的名称）
            [1]: 高基变量对应的列名
            [2]: 这个数据集要测试的分组数 J 的list
    
    time_limit: default 3600
        每个模型最多训练 3600 秒，超过就终止，输出 time_limit_exceeded = True
    '''
    
    save_dir = os.path.join(result_dir,"fold="+str(foldsCount)+"_seed="+str(seed))
    os.makedirs(save_dir,exist_ok=True)

    filename = os.path.join(save_dir, "summary.csv")
    processed = set()
    if os.path.exists(filename):
        result = pd.read_csv(filename)
        for row in range(result.shape[0]):
            exp_key = result.loc[row,'exp_key']
            processed.add(exp_key)
    else:
        result = pd.DataFrame()

    max_J_count = max([len(JLst) for (_,_,JLst) in data_hc_Jlst])

    for idx_J in range(0, max_J_count):
        for (algo_name, clf, other_cat_encoder) in models: 
            for (ds,hc_col,JLst) in loader(data_hc_Jlst):
                if idx_J >= len(JLst): # 该数据集所有 J 已经运行完
                    continue
                J = JLst[idx_J]
                
                rs = ShuffleSplit(n_splits = foldsCount, test_size=.30, random_state=seed)
                for foldIdx, (train_index, test_index) in enumerate(rs.split(ds.X)):
                    for hc_cat_encoder in hc_cat_encoder_list:
                        exp_key = f'{algo_name}_{ds.name}_{hc_col}_{foldIdx}_{J}_{hc_cat_encoder}'
                        print(f"exp_key(algo_name_dsname_hc_fold_J): {exp_key}")
                        if exp_key in processed:
                            print("------===-skip-===----")
                            continue

                        # scale num column; encode cat column; encode hc-cat column
                        X_train,y_train,X_test,y_test = prepare_train_test(ds,
                            train_index, test_index, hc_col, hc_cat_encoder=hc_cat_encoder,
                            hc_group = J, other_cat_encoder = other_cat_encoder, fold=foldIdx, verbose=False)
                        # print("---------------------")
                        # print(f"{X_train.shape = }")
                        # print(f"{y_train.shape = }")
                        # print(f"{X_test.shape = }")
                        # print(f"{y_test.shape = }")    
                        # print("---------------------")
                                                
                        # train a model
                        timeStart = time.time()
                        def train():
                            clf.fit(X_train, y_train)
                        finished,_ = run_with_time_limit(time_limit, train)
                        train_time = time.time() - timeStart
                        
                        if finished:
                            # test performance
                            timeStart = time.time()
                            y_preds = clf.predict(X_test)
                            pred_time = time.time() - timeStart
                            fpr, tpr, _ = metrics.roc_curve(y_test, y_preds)
                            clf_result = {
                                'AUC': metrics.auc(fpr, tpr),
                                'Accuracy': metrics.accuracy_score(y_test, y_preds) ,
                                'Precision': metrics.precision_score(y_test, y_preds), 
                                'Recall': metrics.recall_score(y_test, y_preds),
                                'F1': metrics.f1_score(y_test, y_preds),
                                'Aver_precision': metrics.average_precision_score(y_test, y_preds),
                                'Balanced_accuracy': metrics.balanced_accuracy_score(y_test, y_preds),
                                'train_time(s)': train_time,
                                'pred_time(s)': pred_time
                            }
                            
                            for m in clf_result.keys():
                                new_row = result.shape[0]
                                # print(f"{new_row = }")
                                result.loc[new_row,'exp_key'] = exp_key
                                result.loc[new_row,'ds_name'] = ds.name
                                result.loc[new_row,'train_samples'] = X_train.shape[0]
                                result.loc[new_row,'train_dim'] = X_train.shape[1]
                                result.loc[new_row,'test_samples'] = X_test.shape[0]
                                result.loc[new_row,'test_dim'] = X_train.shape[1]                            
                                result.loc[new_row,'hc_name'] = hc_col
                                # statistics about hc_col                            
                                # for col in ds.stats:
                                #     result.loc[new_row,'hc_'+col] = ds.stats.loc[hc_col,col]
                                result.loc[new_row,'fold'] = foldIdx 
                                result.loc[new_row,'algo_name'] = algo_name
                                result.loc[new_row,'J'] = J
                                result.loc[new_row,'hc_cat_encoder'] = hc_cat_encoder
                                result.loc[new_row,'other_cat_encoder'] = other_cat_encoder
                                result.loc[new_row,'measure'] = m
                                result.loc[new_row,'performance'] = clf_result[m]                                
                        else: # cannot train within time_limit
                            new_row = result.shape[0]
                            # print(f"{new_row = }")
                            result.loc[new_row,'exp_key'] = exp_key
                            result.loc[new_row,'ds_name'] = ds.name
                            result.loc[new_row,'train_samples'] = X_train.shape[0]
                            result.loc[new_row,'train_dim'] = X_train.shape[1]
                            result.loc[new_row,'test_samples'] = X_test.shape[0]
                            result.loc[new_row,'test_dim'] = X_train.shape[1]                            
                            result.loc[new_row,'hc_name'] = hc_col
                            # statistics about hc_col                            
                            # for col in ds.stats:
                            #     result.loc[new_row,'hc_'+col] = ds.stats.loc[hc_col,col]
                            result.loc[new_row,'fold'] = foldIdx 
                            result.loc[new_row,'algo_name'] = algo_name
                            result.loc[new_row,'J'] = J
                            result.loc[new_row,'hc_cat_encoder'] = hc_cat_encoder
                            result.loc[new_row,'other_cat_encoder'] = other_cat_encoder
                            result.loc[new_row,'measure'] = 'time_limit_exceeded'
                            result.loc[new_row,'performance'] = 'True'
                            
                        result.to_csv(filename,index=False)
                        processed.add(exp_key)
    return merge_result(filename, data_hc_Jlst)
 
def merge_result(exp_summary_file, data_hc_Jlst=data_hc_Jlst):
    '''
    Merge statistics of hc_col from datasets into exp results

    Parameters
    ----------
    exp_summary_file : string
        location of exp_summary csv file
    data_hc_Jlst : list(tuple), optional
        list of (dataset, hc_col, JList). The default is data_hc_Jlst.

    Returns
    -------
    result_merged : pandas.DataFrame
        merged table where columns are added to include statistics for hc_col

    '''
    hc_stats = pd.DataFrame()
    for (ds,hc_col,_) in loader(data_hc_Jlst):
        # ds = datasets.load_one_set(ds_name)
        # hc_col = data_hc_Jlst[ds_name][0]
        new_row = hc_stats.shape[0]
        hc_stats.loc[new_row,'ds_name'] = ds.name
        hc_stats.loc[new_row,'hc_name'] = hc_col
        for col in ds.stats:
            hc_stats.loc[new_row,'hc_'+col] = ds.stats.loc[hc_col,col]

    result = pd.read_csv(exp_summary_file)
    result_merged = pd.merge(result, hc_stats, left_on=['ds_name','hc_name'], right_on=['ds_name','hc_name'])
    
    base = os.path.dirname(exp_summary_file)
    result_merged.to_csv(os.path.join(base,'summary_merged.csv'),index=False)
    
    return result_merged
   
 
if __name__ == '__main__': 
    
    
    dirname = '../results/exp22_04_19/'
    fast_algos = get_model_by_name(names=['GBDT', 'DT', 'RF', 'LR']) # about 18 hours
    NeuralNet = get_model_by_name(names=['NeuralNet']) # about 5 days
    SVM = get_model_by_name(names=['RBFSVM'])    # about 15 days (excluding Crime)
    kNN = get_model_by_name(names=['kNN'])       # about 6 hours, unknown pred_time
    tree_algos = get_model_by_name(names=['GBDT', 'DT', 'RF'])
    print(f'{fast_algos=}')
    print(f'{NeuralNet=}')
    print(f'{SVM=}')
    print(f'{kNN=}')

    small_sets = get_data_Jlst_by_name(names=['Adult','Colleges','EmployeeSalaries','TrafficViolations','RoadSafety'])
    H1BVisa = get_data_Jlst_by_name(names=['H1BVisa'])
    AutoLoan = get_data_Jlst_by_name(names=['AutoLoan'])
    Crime = get_data_Jlst_by_name(names=['Crime'])
    print(f'{len(small_sets)=}')    
    print(f'{small_sets=}')    
    print(f'{H1BVisa=}')    
    print(f'{AutoLoan=}')    
    print(f'{Crime=}')    
    print('--------------------')
    
    
    # # On machine 1: fast x all: about 18 hours
    # exp(fast_algos, 5, data_hc_Jlst, seed=0, result_dir= dirname+'fast_algos')
    # # For tree based algos, TargetEncoding is usually a better choice than one-hot
    # # We try MHE+TargetEncoding and compare it with MHE+one-hot
    # exp(tree_algos, 5, data_hc_Jlst, hc_cat_encoder_list = ['mhe+te'], seed=0, result_dir=dirname+'tree_algos_MHE+TE')

    # # On machine 2: baseline, where hc_cat is encoded by one-hot
    Jlst_1 = create_dataset_J_lst(ratio_lst = [1.0])
    # exp(fast_algos, 5, Jlst_1, hc_cat_encoder_list = ['one-hot'], seed=0, result_dir= dirname+'baseline/one-hot_fast_algos')
    # exp(tree_algos, 5, Jlst_1, hc_cat_encoder_list = ['TargetEncoder'], seed=0, result_dir= dirname+'baseline/TargetEncoder_tree_algos')
    # exp(NeuralNet, 5, Jlst_1, hc_cat_encoder_list = ['one-hot'], seed=0, result_dir= dirname+'baseline/one-hot_NeuralNet')
    # exp(kNN, 5, Jlst_1, hc_cat_encoder_list = ['one-hot'], seed=0, result_dir= dirname+'baseline/one-hot_kNN')
    # exp(SVM, 5, Jlst_1, hc_cat_encoder_list = ['one-hot'], seed=0, result_dir= dirname+'baseline/one-hot_SVM')
    dirname = '../results/exp22_05_14/'
    exp(SVM, 5, Jlst_1[6:7], hc_cat_encoder_list = ['one-hot'], seed=0, result_dir= dirname+'baseline/one-hot_SVM',
        time_limit=3600000)


    # # On machine 3: NeuralNet x small_sets + H1BVisa: 44 hr
    # exp(NeuralNet, 5, small_sets, seed=0, result_dir=dirname+'NeuralNet_small_sets') # 27 hr
    # exp(NeuralNet, 5, H1BVisa, seed=0, result_dir=dirname+'NeuralNet_H1BVisa') # 17 hr
    
    # # On machine 4: NeuralNet x AutoLoan: 42 hours
    # exp(NeuralNet, 5, AutoLoan, seed=0, result_dir=dirname+'NeuralNet_AutoLoan') # 42 hr

    # # On machine 5: NeuralNet x Crime: 50 hours
    # exp(NeuralNet, 5, Crime, seed=0, result_dir=dirname+'NeuralNet_Crime') # 50 hr

    # # On machine 6: SVM x small_sets: 44 hr
    # exp(SVM, 5, small_sets, seed=0, result_dir=dirname+'SVM_small_sets') # 44 hr    
    
    # # On machine 7: SVM x H1BVisa: 70 hr
    # exp(SVM, 5, H1BVisa, seed=0, result_dir=dirname+'SVM_H1BVisa') # 70 hr

    # # On machine 8: kNN x all: 70 hr
    # exp(kNN, 5, data_hc_Jlst, seed=0, result_dir=dirname+'kNN')

    # # On machine 9: SVM x AutoLoan: 230 hr
    # exp(SVM, 5, AutoLoan, seed=0, result_dir=dirname+'SVM_AutoLoan') # 230 hr























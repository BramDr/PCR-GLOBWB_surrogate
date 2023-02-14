import os
import os.path

import pandas as pd

def _get_performance_values(values_file: str) -> pd.DataFrame:
    perf = pd.read_csv(values_file)
    try:
        best_index = perf['test_loss'].argmin()
    except KeyError:
        best_index = perf['train_loss'].argmin()
        
    perf_df = perf.iloc[best_index].to_frame().transpose()
    perf_df["epoch"] = perf_df.index
    perf_df = perf_df.reset_index(drop = True)
    
    return perf_df

def _get_performance_hyperparameters(hyperparameter_dir: str) -> pd.DataFrame:
    hyperparameters = os.path.basename(hyperparameter_dir)
    
    keys = []
    values = []
    options = hyperparameters.split("-")
    keydone = False
    for option in options:
        if keydone:
            keyvalue = option.split("_", 1)
            values.append(keyvalue[0])
            
            try:
                keys.append(keyvalue[1])
            except IndexError:
                pass
        else: 
            keys.append(option)
            keydone = True
    
    hyperparameter_dict = {k:[v] for k, v in zip(keys, values)}
    hyperparameter_df = pd.DataFrame.from_dict(hyperparameter_dict)
    
    performance = pd.DataFrame()
    values_files = os.listdir(hyperparameter_dir)
    for values_file in values_files:
        path = "{}/{}".format(hyperparameter_dir, values_file)
        if values_file != "statistics.csv":
            continue
        
        performance_values = _get_performance_values(path)
        performance = pd.concat((performance, performance_values), axis = 0)
    
    performance = pd.concat((hyperparameter_df, performance), axis = 1)
    return performance

def _get_performance_feature(feature_dir: str) -> pd.DataFrame:
    feature = os.path.basename(feature_dir).split("-")[1]
    
    performance = pd.DataFrame()
    hyperparmaters_dirs = os.listdir(feature_dir)
    for hyperparmaters_dir in hyperparmaters_dirs:
        path = "{}/{}".format(feature_dir, hyperparmaters_dir)
        if not os.path.isdir(path):
            continue
        
        performance_hyperparameters = _get_performance_hyperparameters(path)
        performance = pd.concat((performance, performance_hyperparameters), axis = 0)
        
    performance.insert(0, "feature", feature)
    return performance

def _get_performance_sample(sample_dir: str) -> pd.DataFrame:
    sample = os.path.basename(sample_dir).split("-")[1]
    
    performance = pd.DataFrame()
    feature_dirs = os.listdir(sample_dir)
    for feature_dir in feature_dirs:
        path = "{}/{}".format(sample_dir, feature_dir)
        if not os.path.isdir(path):
            continue
        
        performance_feature = _get_performance_feature(path)
        performance = pd.concat((performance, performance_feature), axis = 0)
        
    performance.insert(0, "sample", sample)
    return performance

def evaluate_hyperparameters(root_dir: str) -> pd.DataFrame:
    
    performance = pd.DataFrame()
    feature_dirs = os.listdir(root_dir)
    for feature_dir in feature_dirs:
        path = "{}/{}".format(root_dir, feature_dir)
        if not os.path.isdir(s = path):
            continue
        
        performance_features = _get_performance_feature(path)
        performance = pd.concat(objs = (performance, performance_features), axis = 0)
        
    return performance
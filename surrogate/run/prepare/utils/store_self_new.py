from typing import Sequence
import pathlib as pl
import pickle

import pandas as pd
import numpy as np

from surrogate.utils.data import load_meta
from surrogate.utils.data import load_array

from .subset_meta_array import subset_meta_array
from .constant_meta_array import constant_meta_array

def store_self_new(save_dir: pl.Path,
               features: pd.DataFrame,
               resolution: str,
               name_list: Sequence[np.ndarray],
               samples_list: Sequence[np.ndarray],
               dates_list: Sequence[np.ndarray],
               lats_list: Sequence[np.ndarray],
               lons_list: Sequence[np.ndarray],
               dir_out_list: Sequence[np.ndarray],
               verbose: int = 1):

    for index in features.index:
        source = features["source"].loc[index]
        feature = features["feature"].loc[index]
        section = features["section"].loc[index]
        option = features["option"].loc[index]
        variable = features["variable"].loc[index]
        constant = features["constant"].loc[index]
        
        if verbose > 0:
            print("Working on {}".format(feature))
        
        meta = None
        array = None
        if len(variable) > 0:
            meta_file = pl.Path("{}/{}/{}/{}/{}_meta.pkl".format(save_dir, source, section, option, variable))
            meta = load_meta(file=meta_file,
                             verbose=verbose - 1)
            
            array_file = pl.Path("{}/{}/{}/{}/{}.npy".format(save_dir, source, section, option, variable))
            array = load_array(file=array_file,
                             verbose=verbose - 1)
            
        for name, samples, dates, lats, lons, dir_out in zip(name_list,
                                                             samples_list,
                                                             dates_list,
                                                             lats_list,
                                                             lons_list,
                                                             dir_out_list):
            
            array_out =  pl.Path("{}/{}.npy".format(dir_out, feature))
            meta_out = pl.Path("{}/{}_meta.pkl".format(dir_out, feature))
            
            if array_out.exists() and meta_out.exists():
                continue
            
            if verbose > 0:
                print("Processing {}".format(name))
            
            if array is not None and meta is not None:
                meta_subset, array_subset = subset_meta_array(meta=meta,
                                                              array=array,
                                                              samples=samples,
                                                              dates=dates,
                                                              verbose=verbose - 1)
            else:
                meta_subset, array_subset = constant_meta_array(constant=constant,
                                                                samples=samples,
                                                                lons=lons,
                                                                lats=lats,
                                                                resolution=resolution,
                                                                dates=dates,
                                                                verbose=verbose - 1)
                
            array_out.parent.mkdir(parents=True, exist_ok=True)
            np.save(file=array_out, arr=array_subset)
            
            meta_out.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_out, 'wb') as file:
                pickle.dump(obj=meta_subset, file=file)
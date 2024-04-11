from typing import Sequence
import datetime as dt
import pathlib as pl
import pickle
import re

import pandas as pd
import numpy as np

from surrogate.utils.data import load_meta
from utils.process_spatial_long_subsets import process_spatial_long_subsets
from utils.process_temporal_subsets import process_temporal_subsets
from utils.process_output_subsets import process_output_subsets
from utils.process_subtemporal_subsets import process_subtemporal_subsets
from surrogate.utils.data import load_array
from utils.store_subsets import store_subsets


def store_generated_data(features_df: pd.DataFrame,
                         save_dir: pl.Path,
                         samples_subsets: Sequence[np.ndarray],
                         lons_subsets: Sequence[np.ndarray],
                         lats_subsets: Sequence[np.ndarray],
                         dates_subsets: Sequence[np.ndarray],
                         subdates_subsets: Sequence[np.ndarray],
                         areas_subsets: Sequence[np.ndarray],
                         domains_subsets: Sequence[np.ndarray],
                         out_subsets: Sequence[pl.Path],
                         dataset: str,
                         ldd: pd.DataFrame,
                         verbose: int = 1) -> None:

    sources = features_df["source"].to_list()
    features = features_df["feature"].to_list()
    options = features_df["option"].to_list()

    source = sources[0]
    feature = features[0]
    option = options[0]
    for feature, source, option in zip(features,
                                       sources,
                                       options):
        if source != "generated":
            continue

        if verbose > 0:
            print("Working on {}".format(option))

        origional_feature = feature
        is_upstream = "upstream_" in feature
        if is_upstream:
            feature = re.sub("upstream_", "", feature)

        unique_domains = np.unique(np.concatenate(domains_subsets))
        if len(unique_domains) <= 0:
            raise ValueError("No domains found")

        # Identical for each domain
        domain =  unique_domains[0]
        if domain == "global":
            domain_dir = pl.Path("{}/".format(save_dir))
        else:
            domain_dir = pl.Path("{}/{}".format(save_dir, domain))
        
        date_strings = [dir.stem for dir in domain_dir.iterdir() if dir.is_dir()]
        
        date_string = date_strings[0]
        date_dir = pl.Path("{}/{}".format(domain_dir, date_string))
        meta_file = pl.Path("{}/{}_meta.pkl".format(date_dir, feature))
        meta = load_meta(file=meta_file)

        # Define spatiotemporal values
        value_subdates = meta["subdates"]
        value_dates = np.array([dt.datetime.strptime(date_string, "%Y%m%d") for date_string in date_strings])
        
        dates_subsets_new = []
        for dates in dates_subsets:
            dates = np.array([date for date in dates if date in value_dates])
            dates_subsets_new.append(dates)
        dates_subsets = dates_subsets_new
                
        t_indices_subsets, t_mapping_subsets, t_frequency_subsets, origional_dates_subsets = process_temporal_subsets(dates_subsets=dates_subsets,
                                                                                                                        array_dates=value_dates)
        meta_out_subsets, array_out_subsets, process_flag_subsets = process_output_subsets(out_subsets=out_subsets,
                                                                                           dataset=dataset,
                                                                                           feature=origional_feature)
        
        st_indices_subsets = None
        st_mapping_subsets = None
        st_frequency_subsets = None
        origional_subdates_subsets = None
        if subdates_subsets is not None:
            st_indices_subsets, st_mapping_subsets, st_frequency_subsets, origional_subdates_subsets = process_subtemporal_subsets(subdates_subsets=subdates_subsets,
                                                                                                                                   array_subdates=value_subdates)
                
        # Varying for each domain
        value_lons = None
        value_lats = None
        
        origional_lats_domains_subsets = []
        origional_lons_domains_subsets = []
        s_indices_domains_subsets = []
        s_mapping_domains_subsets = []
        
        domain = unique_domains[0]
        for domain in unique_domains:
            if verbose > 1:
                print("Working on {}".format(domain))

            if domain == "global":
                domain_dir = pl.Path("{}/".format(save_dir))
            else:
                domain_dir = pl.Path("{}/{}".format(save_dir, domain))
            
            date_strings = [dir.stem for dir in domain_dir.iterdir() if dir.is_dir()]
            
            date_string = date_strings[0]
            date_dir = pl.Path("{}/{}".format(domain_dir, date_string))
            meta_file = pl.Path("{}/{}_meta.pkl".format(date_dir, feature))
            meta = load_meta(file=meta_file)
            
            value_lats = meta["lats"]
            value_lons = meta["lons"]
            
            s_indices_subsets, s_mapping_subsets, origional_lons_subsets, origional_lats_subsets = process_spatial_long_subsets(lons_subsets=lons_subsets,
                                                                                                                        lats_subsets=lats_subsets,
                                                                                                                        array_lons=value_lons,
                                                                                                                        array_lats=value_lats,
                                                                                                                        domain=domain,
                                                                                                                        domains_subsets=domains_subsets)

            origional_lats_domains_subsets.append(origional_lats_subsets)
            origional_lons_domains_subsets.append(origional_lons_subsets)
            s_indices_domains_subsets.append(s_indices_subsets)
            s_mapping_domains_subsets.append(s_mapping_subsets)
                
        if np.sum(np.array(process_flag_subsets)) == 0:
            if verbose > 0:
                print("All subset already stored")
            continue
        
        # Loop dates
        t_indices_full = np.concatenate(t_indices_subsets, axis=0)
        t_indices_full = np.unique(t_indices_full)
        
        arrays_domain_subsets = []
        for _ in range(len(unique_domains)):
            subset_arrays = []
            for _ in range(len(process_flag_subsets)):
                time_arrays = []
                for _ in range(len(t_indices_full)):
                    time_arrays.append(None)
                subset_arrays.append(time_arrays)
            arrays_domain_subsets.append(subset_arrays)

        domain_index = 0
        domain = unique_domains[domain_index]
        for domain_index, (domain, s_indices_subsets) in enumerate(zip(unique_domains,
                                                                       s_indices_domains_subsets)):

            if verbose > 1:
                print("Working on {}".format(domain))

            if domain == "global":
                domain_dir = pl.Path("{}/".format(save_dir))
            else:
                domain_dir = pl.Path("{}/{}".format(save_dir, domain))
            
            date_strings = [dir.stem for dir in domain_dir.iterdir() if dir.is_dir()]
            
            for t_index_full in t_indices_full:
                if verbose > 2:
                    print("Processing {}".format(t_index_full))

                date_index = t_index_full                    
                date_string = date_strings[date_index]
                date_dir = pl.Path("{}/{}".format(domain_dir, date_string))

                value_file = pl.Path("{}/{}.npy".format(date_dir, feature))
                value_array = load_array(file=value_file, verbose=verbose - 1)
                value_array = np.squeeze(a=value_array, axis=-1)

                if is_upstream:
                    value_array_upstream = np.zeros_like(value_array)
                    for d_index, d_id in zip(ldd.index, ldd["downstream"]):
                        value_array_upstream[..., d_id] += value_array[..., d_index]
                    value_array = value_array_upstream

                for subset_index, (s_indices, t_indices, st_indices, area, process_flag) in enumerate(zip(s_indices_subsets,
                                                                                                        t_indices_subsets,
                                                                                                        st_indices_subsets,
                                                                                                        areas_subsets,
                                                                                                        process_flag_subsets)):
                    if process_flag and t_index_full in t_indices:
                        subset_array = np.array(value_array[..., s_indices])
                        subset_array = np.array(subset_array[st_indices, ...])
                
                        if feature == "sub_surface_water_storage":
                            subset_array /= area

                        time_index = np.where(t_indices == t_index_full)[0][0]
                        arrays_domain_subsets[domain_index][subset_index][time_index] = subset_array
        
        # Reshape spatial information
        origional_lons_subsets = []
        origional_lats_subsets = []
        s_mapping_subsets = []        
        for subset_index in range(len(process_flag_subsets)):
            
            origional_lons_list = []
            origional_lats_list = []
            s_mapping_list = []  
            for domain_index in range(len(unique_domains)):
                
                origional_lons = origional_lons_domains_subsets[domain_index][subset_index]
                origional_lats = origional_lats_domains_subsets[domain_index][subset_index]
                s_mapping = s_mapping_domains_subsets[domain_index][subset_index]
                if origional_lons is None or origional_lats is None or s_mapping is None:
                    continue  # Skip domains out of spatial domain
                
                origional_lons_list.append(origional_lons)
                origional_lats_list.append(origional_lats)
                s_mapping_list.append(s_mapping)
            
            origional_lons = np.concatenate(origional_lons_list, axis=0)
            origional_lats = np.concatenate(origional_lats_list, axis=0)
            
            s_mapping_new = []
            prev_s_mapping_max = 0
            for s_mapping in s_mapping_list:
                s_mapping += prev_s_mapping_max
                prev_s_mapping_max = np.max(s_mapping) + 1
                s_mapping_new.append(s_mapping)
            s_mapping = np.concatenate(s_mapping_new, axis=0)
            
            origional_lons_subsets.append(origional_lons)
            origional_lats_subsets.append(origional_lats)
            s_mapping_subsets.append(s_mapping)
            
        arrays_subsets = []
        for _ in range(len(process_flag_subsets)):
            time_arrays = []
            for _ in range(len(t_indices_full)):
                time_arrays.append(None)
            arrays_subsets.append(time_arrays)
        
        for subset_index in range(len(process_flag_subsets)):
            for time_index in range(len(t_indices_full)):
                
                arrays_list = []
                for domain_index in range(len(unique_domains)):
                    array = arrays_domain_subsets[domain_index][subset_index][time_index]
                    if array is None:
                        continue
                    
                    arrays_list.append(array)

                if len(arrays_list) > 0:
                    array = np.concatenate(arrays_list, axis=0)
                    arrays_subsets[subset_index][time_index] = array
                
        store_subsets(arrays_subsets = arrays_subsets,
                    samples_subsets = samples_subsets,
                    lons_subsets =lons_subsets,
                    lats_subsets = lats_subsets,
                    dates_subsets = dates_subsets,
                    subdates_subsets=subdates_subsets,
                    origional_lons_subsets = origional_lons_subsets,
                    origional_lats_subsets = origional_lats_subsets,
                    origional_dates_subsets = origional_dates_subsets,
                    origional_subdates_subsets=origional_subdates_subsets,
                    s_mapping_subsets = s_mapping_subsets,
                    t_mapping_subsets = t_mapping_subsets,
                    st_mapping_subsets=st_mapping_subsets,
                    meta_out_subsets = meta_out_subsets,
                    array_out_subsets = array_out_subsets,
                    process_flag_subsets = process_flag_subsets)

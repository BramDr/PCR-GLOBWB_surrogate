from typing import Sequence, Optional, Union
import datetime as dt
import pathlib as pl
import configparser as cp
import re
# import time

import pandas as pd
import numpy as np
import pcraster as pcr

from .get_pcr_coordinates import get_pcr_coordinates
from .get_nc_coordinates_dates import get_nc_coordinates_dates
from .process_spatial_subsets import process_spatial_subsets
from .process_dates_subsets import process_dates_subsets
from .process_output_subsets import process_output_subsets
from .get_pcr_array import get_pcr_array
from .get_nc_array import get_nc_array
from .store_subsets import store_subsets
from .store_subsets_combined import store_subsets_combined


def store_input_data(features_df: pd.DataFrame,
                     configuration: cp.RawConfigParser,
                     samples_subsets: Sequence[np.ndarray],
                     lons_subsets: Sequence[np.ndarray],
                     lats_subsets: Sequence[np.ndarray],
                     dates_subsets: Union[Sequence[np.ndarray], np.ndarray],
                     out_subsets: Sequence[pl.Path],
                     dataset: str,
                     combine: bool = False,
                     name_subsets: Optional[Sequence[str]] = None,
                     arrays_additional: Optional[dict[str, np.ndarray]] = None,
                     first_date: dt.datetime = dt.datetime(year = 2000,
                                                           month = 1,
                                                           day = 1,
                                                           hour=0,
                                                           minute=0,
                                                           second=0,
                                                           microsecond=0),
                     waterbody_ids_file: Optional[pl.Path] = None,
                     force_output: bool = False,
                     verbose: int = 1) -> None:

    input_prefix = configuration.get(section="globalOptions",
                                     option="inputdir")
    input_prefix = pl.Path(input_prefix)
        
    sources = features_df["source"].to_list()
    features = features_df["feature"].to_list()
    sections = ["" for _ in features]
    options = ["" for _ in features]
    levels = ["" for _ in features]
    variables = ["" for _ in features]
    onehots = ["" for _ in features]
    bumps = ["" for _ in features]
    aggregations = ["" for _ in features]
    
    if "section" in features_df.columns:
        sections = features_df["section"].to_list()
    if "option" in features_df.columns:
        options = features_df["option"].to_list()
    if "level" in features_df.columns:
        levels = features_df["level"].to_list()
    if "variable" in features_df.columns:
        variables = features_df["variable"].to_list()
    if "onehot" in features_df.columns:
        onehots = features_df["onehot"].to_list()
    if "bump" in features_df.columns:
        bumps = features_df["bump"].to_list()
    if "aggregation" in features_df.columns:
        aggregations = features_df["aggregation"].to_list()
        
    waterbody_ids = None
    if waterbody_ids_file is not None:
        pcr.setclone(str(waterbody_ids_file))
        waterbody_ids = pcr.readmap(str(waterbody_ids_file))
        waterbody_ids = pcr.cover(waterbody_ids, 0)
    
    for feature, source, section, option, level, variable, onehot, bump, aggregation in zip(features,
                                                                                          sources,
                                                                                          sections,
                                                                                          options,
                                                                                          levels,
                                                                                          variables,
                                                                                          onehots,
                                                                                          bumps,
                                                                                          aggregations):
        if source != "input":
            continue

        if verbose > 0:
            print("Working on {} ({} {})".format(feature, section, option))
            
        if (aggregation != "" and aggregation != "none") and waterbody_ids is None:
            raise ValueError("Aggregation is {} but waterbody_ids is None".format(aggregation))
        
        # print(f"Processing {len(out_subsets)} subsets")
        
        meta_out_subsets, array_out_subsets, process_flag_subsets = process_output_subsets(out_subsets=out_subsets,
                                                                                           dataset=dataset,
                                                                                           feature=feature,
                                                                                           combine=combine)
                
        if force_output:
            process_flag_subsets = [True for _ in process_flag_subsets]

        if np.sum(np.array(process_flag_subsets)) == 0:
            if verbose > 0:
                print("All subset already stored")
            continue

        if option.isnumeric():
            value = option
        else:
            value = configuration.get(section=section,
                                    option=option)

        value_file = pl.Path("{}/{}".format(input_prefix,
                                            value))
        if value.startswith("/"):
            value_file = pl.Path(value)
        if level != "":
            value_file_name = re.sub("dzRel%04d", level, value_file.name)
            value_file = pl.Path(
                "{}/{}".format(value_file.parent,
                                    value_file_name))

        # counter_start = time.perf_counter()
        value_lats = None
        value_lons = None
        value_datetimes = None
        if value.endswith((".map")) and value_file is not None:
            value_lons, value_lats = get_pcr_coordinates(file=value_file)
        elif value.endswith((".nc", ".nc4")) and value_file is not None:
            value_lons, value_lats, value_datetimes = get_nc_coordinates_dates(file=value_file)
        # counter_end = time.perf_counter()
        # print(f"Finished get_pcr_coordinates(_dates) in {counter_end - counter_start} seconds")
        # print(f"Aquired {len(value_lons)} lons and {len(value_lats)} lats")x
        
        # counter_start = time.perf_counter()
        _, s_indices_subsets, s_mapping_subsets, origional_lons_subsets, origional_lats_subsets = process_spatial_subsets(lons_subsets=lons_subsets,
                                                                                                                          lats_subsets=lats_subsets,
                                                                                                                          array_lons=value_lons,
                                                                                                                          array_lats=value_lats)
        # counter_end = time.perf_counter()
        # print(f"Finished process_spatial_subsets in {counter_end - counter_start} seconds")
                
        value_dates = None
        if value_datetimes is not None:
            value_dates = np.array([datetime.date() for datetime in value_datetimes])
            value_dates = np.unique(value_dates)
            
        # counter_start = time.perf_counter()
        if isinstance(dates_subsets, np.ndarray):
            _, d_indices_subsets, d_mapping_subsets, _, origional_dates_subsets = process_dates_subsets(dates_subsets=[dates_subsets],
                                                                                                        array_dates=value_dates)
            storage_dates_subsets = [dates_subsets] * len(samples_subsets)
            d_indices_subsets = d_indices_subsets * len(samples_subsets)
            d_mapping_subsets = d_mapping_subsets * len(samples_subsets)
            origional_dates_subsets = origional_dates_subsets * len(samples_subsets)
        elif isinstance(dates_subsets, Sequence):
            _, d_indices_subsets, d_mapping_subsets, _, origional_dates_subsets = process_dates_subsets(dates_subsets=dates_subsets,
                                                                                                        array_dates=value_dates)
            storage_dates_subsets = dates_subsets
        else:
            raise ValueError("dates_subsets is not a Sequence or np.ndarray")
                
        # counter_end = time.perf_counter()
        # print(f"Finished process_dates_subset Sequence in {counter_end - counter_start} seconds")
        
        # Loop dates
        if d_indices_subsets is None:
            d_indices_subsets = [np.array(None)]
            
        total_d_indices = np.concatenate(d_indices_subsets, axis=0)
        total_d_indices = np.unique(total_d_indices)

        arrays_subsets = []
        for _ in enumerate(process_flag_subsets):
            time_arrays = []
            for _ in enumerate(total_d_indices):
                time_arrays.append(None)
            arrays_subsets.append(time_arrays)

        # get_array_counter_total = 0
        # subset_index_counter_total = 0
        
        total_d_index = total_d_indices[0]
        for total_d_index in total_d_indices:
            if verbose > 1:
                print("Processing {}".format(total_d_index))
                
            date_index = total_d_index
            if value_dates is None:
                date_index = None

            # counter_start = time.perf_counter()
            if value.endswith((".map")) and value_file is not None:
                value_array = get_pcr_array(file=value_file,
                                            aggregate=aggregation,
                                            aggregate_ids=waterbody_ids)
            elif value.endswith((".nc", ".nc4")) and value_file is not None:
                value_array = get_nc_array(file=value_file,
                                           variable_name=variable,
                                           date_index=date_index,
                                           aggregate=aggregation,
                                           aggregate_ids=waterbody_ids)
            else:
                value_array = np.array([float(value)],
                                       dtype=np.float32)
            # counter_end = time.perf_counter()
            # get_array_counter_total += counter_end - counter_start

            # counter_start = time.perf_counter()
            for subset_index, (s_indices, d_indices, process_flag) in enumerate(zip(s_indices_subsets,
                                                                                     d_indices_subsets,
                                                                                     process_flag_subsets)):                
                if process_flag and s_indices is not None and total_d_index in d_indices:                    
                    subset_array = np.array(value_array[s_indices])
                    if onehot != "":
                        subset_array = subset_array == float(onehot)
                        
                    time_index = np.where(d_indices == total_d_index)[0][0]
                    arrays_subsets[subset_index][time_index] = subset_array
            # counter_end = time.perf_counter()
            # subset_index_counter_total += counter_end - counter_start
        # print(f"Finished time loop with get_array in {get_array_counter_total} seconds and subset_index in {subset_index_counter_total} seconds")
                                
        if bump == "first":
            for subset_index, (dates, d_mapping) in enumerate(zip(storage_dates_subsets,
                                                                  d_mapping_subsets)):                
                if np.sum(d_mapping) > 0:
                    raise ValueError("Bump first not possible with multiple dates")
                
                if first_date in dates:
                    d_mapping[1:] = 1
                else:             
                    d_mapping[:] = 1
                    
                d_mapping_subsets[subset_index] = d_mapping
                
                subset_array = arrays_subsets[subset_index][0]
                if subset_array is not None:
                    arrays_subsets[subset_index].append(subset_array * 0)
        
        # counter_start = time.perf_counter()
        if not combine:
            store_subsets(arrays_subsets = arrays_subsets,
                        samples_subsets = samples_subsets,
                        lons_subsets =lons_subsets,
                        lats_subsets = lats_subsets,
                        dates_subsets = storage_dates_subsets,
                        origional_lons_subsets = origional_lons_subsets,
                        origional_lats_subsets = origional_lats_subsets,
                        origional_dates_subsets = origional_dates_subsets,
                        s_mapping_subsets = s_mapping_subsets,
                        d_mapping_subsets = d_mapping_subsets,
                        meta_out_subsets = meta_out_subsets,
                        array_out_subsets = array_out_subsets,
                        process_flag_subsets = process_flag_subsets)
        else:
            if name_subsets is None:
                raise ValueError("name_subsets is None")
            
            store_subsets_combined(names = name_subsets,
                                   arrays_subsets = arrays_subsets,
                                   arrays_additional = arrays_additional,
                                   samples_subsets = samples_subsets,
                                   lons_subsets = lons_subsets,
                                   lats_subsets = lats_subsets,
                                   dates_subsets = storage_dates_subsets,
                                   origional_lons_subsets = origional_lons_subsets,
                                   origional_lats_subsets = origional_lats_subsets,
                                   origional_dates_subsets = origional_dates_subsets,
                                   s_mapping_subsets = s_mapping_subsets,
                                   d_mapping_subsets = d_mapping_subsets,
                                   meta_out_subsets = meta_out_subsets,
                                   array_out_subsets = array_out_subsets,
                                   process_flag_subsets = process_flag_subsets)
        # counter_end = time.perf_counter()
        # print(f"Finished store_subsets(_combined) in {counter_end - counter_start} seconds")


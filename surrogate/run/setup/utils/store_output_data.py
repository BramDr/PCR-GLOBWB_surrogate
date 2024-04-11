from typing import Sequence, Optional, Union
import pathlib as pl
import configparser as cp

import pandas as pd
import numpy as np
import pcraster as pcr

from .get_nc_coordinates_dates import get_nc_coordinates_dates
from .process_spatial_subsets import process_spatial_subsets
from .process_dates_subsets import process_dates_subsets
from .process_dates_subsets import process_dates_subset
from .process_output_subsets import process_output_subsets
from .get_nc_array import get_nc_array
from .store_subsets import store_subsets
from .store_subsets_combined import store_subsets_combined


def store_output_data(features_df: pd.DataFrame,
                      configuration: cp.RawConfigParser,
                      domain: str,
                      samples_subsets: Sequence[np.ndarray],
                      lons_subsets: Sequence[np.ndarray],
                      lats_subsets: Sequence[np.ndarray],
                      dates_subsets: Union[Sequence[np.ndarray], np.ndarray],
                      out_subsets: Sequence[pl.Path],
                      dataset: str,
                      combine: bool = False,
                      name_subsets: Optional[Sequence[str]] = None,
                      arrays_additional: Optional[dict[str, np.ndarray]] = None,
                      output_postfix: str = "netcdf",
                      waterbody_ids_file: Optional[pl.Path] = None,
                      area_file: Optional[pl.Path] = None,
                      force_output: bool = False,
                      verbose: int = 1) -> None:

    output_prefix = configuration.get(section="globalOptions",
                                      option="outputdir")
    output_prefix = pl.Path(output_prefix)

    sources = features_df["source"].to_list()
    features = features_df["feature"].to_list()
    options = ["" for _ in features]
    variables = ["" for _ in features]
    aggregations = ["" for _ in features]
    conversions = ["" for _ in features]
    
    if "option" in features_df.columns:
        options = features_df["option"].to_list()
    if "variable" in features_df.columns:
        variables = features_df["variable"].to_list()
    if "aggregation" in features_df.columns:
        aggregations = features_df["aggregation"].to_list()
    if "conversion" in features_df.columns:
        conversions = features_df["conversion"].to_list()
        
    waterbody_ids = None
    if waterbody_ids_file is not None:
        pcr.setclone(str(waterbody_ids_file))
        waterbody_ids = pcr.readmap(str(waterbody_ids_file))
        waterbody_ids = pcr.cover(waterbody_ids, 0)
        
    area = None
    if area_file is not None:
        pcr.setclone(str(area_file))
        area = pcr.readmap(str(area_file))
        area = pcr.cover(area, 0)
    
    for feature, source, option, variable, aggregation, conversion in zip(features,
                                                                          sources,
                                                                          options,
                                                                          variables,
                                                                          aggregations,
                                                                          conversions):
        if source != "output":
            continue

        if verbose > 0:
            print("Working on {} ({})".format(feature, option))
        
        if (aggregation != "" and aggregation != "none") and waterbody_ids is None:
            raise ValueError("Aggregation is {} but waterbody_ids is None".format(aggregation))
        
        if (conversion != "" and conversion != "none") and area is None:
            raise ValueError("Conversion is {} but area is None".format(conversion))
                
        meta_out_subsets, array_out_subsets, process_flag_subsets = process_output_subsets(out_subsets=out_subsets,
                                                                                           dataset=dataset,
                                                                                           domain=domain,
                                                                                           feature=feature,
                                                                                           combine=combine)
                
        if force_output:
            process_flag_subsets = [True for _ in process_flag_subsets]
            
        if np.sum(np.array(process_flag_subsets)) == 0:
            if verbose > 0:
                print("All subset already stored")
            continue

        if domain == "global":
            value_dir = pl.Path("{}/{}".format(output_prefix,
                                                output_postfix))
        else:
            value_dir = pl.Path("{}/{}/{}".format(output_prefix.parent,
                                                        domain,
                                                        output_postfix))
        
        value_file = [file for file in value_dir.rglob("{}_*.nc".format(option))][0]
        
        value_lons, value_lats, value_datetimes = get_nc_coordinates_dates(file=value_file)
        
        _, s_indices_subsets, s_mapping_subsets, origional_lons_subsets, origional_lats_subsets = process_spatial_subsets(lons_subsets=lons_subsets,
                                                                                                                          lats_subsets=lats_subsets,
                                                                                                                          array_lons=value_lons,
                                                                                                                          array_lats=value_lats)
        
        if value_datetimes is None:
            raise ValueError("No datetimes found")
        value_dates = np.array([datetime.date() for datetime in value_datetimes])
        value_dates = np.unique(value_dates)
        
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
        
        if d_indices_subsets is None or d_mapping_subsets is None or dates_subsets is None or origional_dates_subsets is None:
            raise ValueError("d_indices_subsets, d_mapping_subsets, dates_subsets or origional_dates_subsets is None")
                
        # Loop dates
        total_d_indices = np.concatenate(d_indices_subsets, axis=0)
        total_d_indices = np.unique(total_d_indices)

        arrays_subsets = []
        for _ in enumerate(process_flag_subsets):
            time_arrays = []
            for _ in enumerate(total_d_indices):
                time_arrays.append(None)
            arrays_subsets.append(time_arrays)

        for total_d_index in total_d_indices:
            if verbose > 1:
                print("Processing {}".format(total_d_index))
            
            date_index = total_d_index                    
            value_array = get_nc_array(file=value_file,
                                        variable_name=variable,
                                        date_index=date_index,
                                        aggregate=aggregation,
                                        aggregate_ids=waterbody_ids,
                                        conversion=conversion,
                                        area=area)

            for subset_index, (s_indices, d_indices, process_flag) in enumerate(zip(s_indices_subsets,
                                                                                    d_indices_subsets,
                                                                                    process_flag_subsets)):
                if process_flag and s_indices is not None and total_d_index in d_indices:
                    subset_array = np.array(value_array[s_indices])
                    time_index = np.where(d_indices == total_d_index)[0][0]
                    arrays_subsets[subset_index][time_index] = subset_array
        
        if not combine:
            store_subsets(arrays_subsets = arrays_subsets,
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
            

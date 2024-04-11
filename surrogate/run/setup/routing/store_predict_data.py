import pathlib as pl
import configparser as cp

import pandas as pd
import numpy as np

from surrogate.run.setup.utils.store_input_data import store_input_data
from surrogate.run.setup.utils.store_output_data import store_output_data

save_dir = pl.Path("./saves")
input_dir = pl.Path("./input")
configuration_dir = pl.Path("../../../../PCR-GLOBWB/configuration")
prepare_base_dir = pl.Path("../../prepare/saves")
prepare_dir = pl.Path("../../prepare/saves/predict")
out_dir = pl.Path("./saves/predict")

resolutions = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir() and dir.stem != "mulres"]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))
    
    save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
    input_resolution_dir = pl.Path("{}/{}".format(input_dir, resolution))
    prepare_base_resolution_dir = pl.Path("{}/{}".format(prepare_base_dir, resolution))
    prepare_resolution_dir = pl.Path("{}/{}".format(prepare_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

    in_features_file = pl.Path("{}/features_input.csv".format(input_resolution_dir))
    in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
    out_features_file = pl.Path("{}/features_output.csv".format(input_resolution_dir))
    out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")
    
    waterbody_ids_file = pl.Path("{}/waterbody_id.map".format(prepare_base_resolution_dir))
    area_file = pl.Path("{}/cell_area.map".format(prepare_base_resolution_dir))

    configuration_file = pl.Path("{}/global_{}.ini".format(configuration_dir, resolution))
    configuration = cp.RawConfigParser()
    configuration.read(configuration_file)

    output_postfix="netcdf"

    domains = [dir.stem for dir in prepare_resolution_dir.iterdir() if dir.is_dir()]

    subset_samples_list = []
    subset_lons_list = []
    subset_lats_list = []
    subset_outs = []
    subset_names = []
    
    domain = domains[0]
    for domain in domains:
        prepare_domain_dir = pl.Path("{}/{}".format(prepare_resolution_dir, domain))
        out_domain_dir = pl.Path("{}/{}".format(out_resolution_dir, domain))
        
        cells_file = pl.Path("{}/cells.parquet".format(prepare_domain_dir))
        cells = pd.read_parquet(cells_file)
        
        dates = pd.date_range(start=cells["start"].iloc[0],
                              end=cells["end"].iloc[0],
                              freq="D").to_pydatetime()
        dates = np.array([datum.date() for datum in dates])
            
        for sequence in np.unique(cells["sequence"]):
            sequence_cells = cells.loc[cells["sequence"] == sequence]
            
            sequence_cells = sequence_cells.sort_index()
            samples = sequence_cells.index.to_numpy()
            lons = sequence_cells["lon"].to_numpy()
            lats = sequence_cells["lat"].to_numpy()
            
            subset_samples_list.append(samples)
            subset_lons_list.append(lons)
            subset_lats_list.append(lats)
            subset_outs.append(out_domain_dir)
            subset_names.append(str(sequence))

    store_input_data(features_df=in_features,
                     configuration=configuration,
                     samples_subsets=subset_samples_list,
                     lons_subsets=subset_lons_list,
                     lats_subsets=subset_lats_list,
                     dates_subsets=dates,
                     out_subsets=subset_outs,
                     dataset="input",
                     waterbody_ids_file=waterbody_ids_file,
                     combine = True,
                     name_subsets = subset_names)

    store_input_data(features_df=out_features,
                    configuration=configuration,
                    samples_subsets=subset_samples_list,
                    lons_subsets=subset_lons_list,
                    lats_subsets=subset_lats_list,
                    dates_subsets=dates,
                    out_subsets=subset_outs,
                    dataset="output",
                    waterbody_ids_file=waterbody_ids_file,
                    combine = True,
                    name_subsets = subset_names)
    
    domain = domains[0]
    for domain in domains:
        print("\tDomain: {}".format(domain))
        
        save_domain_dir = pl.Path("{}/{}".format(save_resolution_dir, domain))
        prepare_domain_dir = pl.Path("{}/{}".format(prepare_resolution_dir, domain))
        out_domain_dir = pl.Path("{}/{}".format(out_resolution_dir, domain))
    
        waterbody_ids_domain_file = pl.Path("{}/waterbody_id.map".format(save_domain_dir))
        area_domain_file = pl.Path("{}/cell_area.map".format(save_domain_dir))
        
        cells_file = pl.Path("{}/cells.parquet".format(prepare_domain_dir))
        cells = pd.read_parquet(cells_file)
        
        dates = pd.date_range(start=cells["start"].iloc[0],
                              end=cells["end"].iloc[0],
                              freq="D").to_pydatetime()
        dates = np.array([datum.date() for datum in dates])
            
        subset_samples_list = []
        subset_lons_list = []
        subset_lats_list = []
        subset_outs = []
        subset_names = []
            
        for sequence in np.unique(cells["sequence"]):
            sequence_cells = cells.loc[cells["sequence"] == sequence]
            
            sequence_cells = sequence_cells.sort_index()
            samples = sequence_cells.index.to_numpy()
            lons = sequence_cells["lon"].to_numpy()
            lats = sequence_cells["lat"].to_numpy()
            
            subset_samples_list.append(samples)
            subset_lons_list.append(lons)
            subset_lats_list.append(lats)
            subset_outs.append(out_domain_dir)
            subset_names.append(str(sequence))
            
        store_output_data(features_df=in_features,
                        configuration=configuration,
                        domain=domain,
                        samples_subsets=subset_samples_list,
                        lons_subsets=subset_lons_list,
                        lats_subsets=subset_lats_list,
                        dates_subsets=dates,
                        out_subsets=subset_outs,
                        output_postfix=output_postfix,
                        dataset="input",
                        waterbody_ids_file=waterbody_ids_domain_file,
                        area_file=area_domain_file,
                        combine = True,
                        name_subsets = subset_names)

        store_output_data(features_df=out_features,
                        configuration=configuration,
                        domain=domain,
                        samples_subsets=subset_samples_list,
                        lons_subsets=subset_lons_list,
                        lats_subsets=subset_lats_list,
                        dates_subsets=dates,
                        out_subsets=subset_outs,
                        output_postfix=output_postfix,
                        dataset="output",
                         waterbody_ids_file=waterbody_ids_domain_file,
                        area_file=area_domain_file,
                        combine = True,
                        name_subsets = subset_names)

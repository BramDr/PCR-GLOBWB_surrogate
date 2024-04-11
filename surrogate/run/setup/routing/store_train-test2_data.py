import pathlib as pl
import configparser as cp

import pandas as pd
import numpy as np

from surrogate.run.setup.utils.store_input_data import store_input_data
from surrogate.run.setup.utils.store_output_data import store_output_data

input_dir = pl.Path("./input")
save_dir = pl.Path("./saves")
configuration_dir = pl.Path("../../../../PCR-GLOBWB/configuration")
prepare_base_dir = pl.Path("../../prepare/saves")
prepare_dir = pl.Path("../../prepare/routing/saves/train-test2")
out_dir = pl.Path("./saves/train-test2")

resolutions = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir() and dir.stem != "mulres"]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))
    
    input_resolution_dir = pl.Path("{}/{}".format(input_dir, resolution))
    save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
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

    domains_list = []
    trainset_samples_list = []
    trainset_lons_list = []
    trainset_lats_list = []
    trainset_dates_list = []
    trainset_outs = []

    routing_types = [dir.stem for dir in prepare_resolution_dir.iterdir() if dir.is_dir()]

    routing_type = routing_types[0]
    for routing_type in routing_types:
        prepare_routing_dir = pl.Path("{}/{}".format(prepare_resolution_dir, routing_type))
        out_routing_dir = pl.Path("{}/{}".format(out_resolution_dir, routing_type))
        
        trainsets = [dir.stem for dir in prepare_routing_dir.iterdir() if dir.is_dir()]
        
        trainset = trainsets[0]
        for trainset in trainsets:
            prepare_trainset_dir = pl.Path("{}/{}".format(prepare_routing_dir, trainset))
            out_trainset_dir = pl.Path("{}/{}".format(out_routing_dir, trainset))

            cells_file = pl.Path("{}/cells.parquet".format(prepare_trainset_dir))
            cells = pd.read_parquet(cells_file)

            samples = cells.index.to_numpy()
            lons = cells["lon"].to_numpy()
            lats = cells["lat"].to_numpy()
            dates = pd.date_range(start=cells["start"].iloc[0],
                                end=cells["end"].iloc[0],
                                freq="D").to_pydatetime()
            dates = np.array([datum.date() for datum in dates])

            trainset_samples_list.append(samples)
            trainset_lons_list.append(lons)
            trainset_lats_list.append(lats)
            trainset_dates_list.append(dates)
            trainset_outs.append(out_trainset_dir)
        
            domains = cells["domain"].to_numpy()
            domains_list.append(domains)
    domains = np.unique(np.concatenate(domains_list))
    
    store_input_data(features_df=in_features,
                    configuration=configuration,
                    samples_subsets=trainset_samples_list,
                    lons_subsets=trainset_lons_list,
                    lats_subsets=trainset_lats_list,
                    dates_subsets=trainset_dates_list,
                    out_subsets=trainset_outs,
                    waterbody_ids_file=waterbody_ids_file,
                    dataset="input")

    store_input_data(features_df=out_features,
                    configuration=configuration,
                    samples_subsets=trainset_samples_list,
                    lons_subsets=trainset_lons_list,
                    lats_subsets=trainset_lats_list,
                    dates_subsets=trainset_dates_list,
                    out_subsets=trainset_outs,
                    waterbody_ids_file=waterbody_ids_file,
                    dataset="output")
    
    domain = domains[0]
    for domain in domains:
        print("\tDomain: {}".format(domain))
        
        save_domain_dir = pl.Path("{}/{}".format(save_resolution_dir, domain))
    
        waterbody_ids_domain_file = pl.Path("{}/waterbody_id.map".format(save_domain_dir))
        area_domain_file = pl.Path("{}/cell_area.map".format(save_domain_dir))
        
        trainset_samples_list = []
        trainset_lons_list = []
        trainset_lats_list = []
        trainset_dates_list = []
        trainset_outs = []

        routing_types = [dir.stem for dir in prepare_resolution_dir.iterdir() if dir.is_dir()]

        routing_type = routing_types[0]
        for routing_type in routing_types:
            
            prepare_routing_dir = pl.Path("{}/{}".format(prepare_resolution_dir, routing_type))
            out_routing_dir = pl.Path("{}/{}".format(out_resolution_dir, routing_type))
            
            trainsets = [dir.stem for dir in prepare_routing_dir.iterdir() if dir.is_dir()]
            
            trainset = trainsets[0]
            for trainset in trainsets:
                prepare_trainset_dir = pl.Path("{}/{}".format(prepare_routing_dir, trainset))
                out_trainset_dir = pl.Path("{}/{}".format(out_routing_dir, trainset))

                cells_file = pl.Path("{}/cells.parquet".format(prepare_trainset_dir))
                cells = pd.read_parquet(cells_file)
                cells = cells.loc[cells["domain"] == domain]
                
                samples = np.array([])
                lons = np.array([])
                lats = np.array([])
                dates = np.array([])
                if cells.index.size > 0:
                    samples = cells.index.to_numpy()
                    lons = cells["lon"].to_numpy()
                    lats = cells["lat"].to_numpy()
                    dates = pd.date_range(start=cells["start"].iloc[0],
                                        end=cells["end"].iloc[0],
                                        freq="D").to_pydatetime()
                    dates = np.array([datum.date() for datum in dates])

                trainset_samples_list.append(samples)
                trainset_lons_list.append(lons)
                trainset_lats_list.append(lats)
                trainset_dates_list.append(dates)
                trainset_outs.append(out_trainset_dir)

        store_output_data(features_df=in_features,
                        configuration=configuration,
                        domain=domain,
                        samples_subsets=trainset_samples_list,
                        lons_subsets=trainset_lons_list,
                        lats_subsets=trainset_lats_list,
                        dates_subsets=trainset_dates_list,
                        out_subsets=trainset_outs,
                        output_postfix=output_postfix,
                        waterbody_ids_file=waterbody_ids_domain_file,
                        area_file=area_domain_file,
                        dataset="input")

        store_output_data(features_df=out_features,
                        configuration=configuration,
                        domain=domain,
                        samples_subsets=trainset_samples_list,
                        lons_subsets=trainset_lons_list,
                        lats_subsets=trainset_lats_list,
                        dates_subsets=trainset_dates_list,
                        out_subsets=trainset_outs,
                        output_postfix=output_postfix,
                        waterbody_ids_file=waterbody_ids_domain_file,
                        area_file=area_domain_file,
                        dataset="output")

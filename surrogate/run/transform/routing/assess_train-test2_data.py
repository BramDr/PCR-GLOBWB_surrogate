import pathlib as pl
import warnings
import re

import pandas as pd
import numpy as np
import scipy.stats as stats

from surrogate.utils.data import load_amt_dataset

setup_dir = pl.Path("../../setup/routing/saves/train-test2")
out_dir = pl.Path("./saves/train-test2")
trainset = "train"
transformer_types = ["default", "log10-sqrt"]

resolutions = [dir.stem for dir in setup_dir.iterdir() if dir.is_dir()]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))

    setup_resolution_dir = pl.Path("{}/{}".format(setup_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

    routing_types = [dir.stem for dir in setup_resolution_dir.iterdir() if dir.is_dir()]

    routing_type = routing_types[0]
    for routing_type in routing_types:
        print("\tRouting type: {}".format(routing_type))
        
        setup_routing_dir = pl.Path("{}/{}".format(setup_resolution_dir, routing_type))
        out_routing_dir = pl.Path("{}/{}".format(out_resolution_dir, routing_type))
    
        setup_trainset_dir = pl.Path("{}/{}".format(setup_routing_dir, trainset))
        
        datasets = [dir.stem for dir in setup_trainset_dir.iterdir() if dir.is_dir()]
        # datasets = ["input"]

        dataset = datasets[0]
        for dataset in datasets:
            print("\t\tDataset: {}".format(dataset))

            setup_dataset_dir = pl.Path("{}/{}".format(setup_trainset_dir, dataset))
            out_dataset_dir = pl.Path("{}/{}".format(out_routing_dir, dataset))
        
            skew_out = pl.Path("{}_skew.csv".format(out_dataset_dir))
            if skew_out.exists():
                print("Already exists")
                continue
            
            features = np.unique([file.stem for file in setup_dataset_dir.rglob("*.npy") if file.is_file()])
            
            skew_dfs = []
            
            feature = features[0]
            for feature in features:
                print("\t\t\tFeature: {}".format(feature))

                name = feature
                name = re.sub("meteo_", "", name)
                name = re.sub("forest_", "", name)
                name = re.sub("grassland_", "", name)
                name = re.sub("irrNonPaddy_", "", name)
                name = re.sub("irrPaddy_", "", name)
                name = re.sub("landSurface_", "", name)
                name = re.sub("groundwater_", "", name)
                name = re.sub("routing_", "", name)
                
                array, meta, _ = load_amt_dataset(feature=feature,
                                                  array_dir=setup_dataset_dir,
                                                  meta_dir=setup_dataset_dir,
                                                  verbose = 0)
                
                array = array.flatten()
                array = np.sort(array)
                array = array - array[0]

                skews = []
                transformers = []
            
                transformer_type = transformer_types[0]
                for transformer_type in transformer_types:       
                    if transformer_type == "default":
                        array_transformed = array
                    elif transformer_type == "log10-sqrt":
                        array_transformed = np.log10(np.sqrt(array) + 1)
                    else:
                        raise ValueError("Unkown type {}".format(transformer_type))
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        skew = np.abs(stats.skew(a = array_transformed))
                    
                    skews.append(skew)
                    transformers.append(transformer_type)
            
                skew_df = {"skew": skews,
                           "transformer": transformers}
                skew_df = pd.DataFrame(skew_df)
                
                skew_df["feature"] = feature
                skew_df["name"] = name
                
                skew_dfs.append(skew_df)
            skew_df = pd.concat(skew_dfs)
            
            skew_df = pd.pivot(skew_df,
                                columns = ["transformer"],
                                values=["skew"],
                                index = ["feature", "name"])
            skew_df.columns = ['_'.join(str(s).strip() for s in col if s) for col in skew_df.columns]
            skew_df = skew_df.reset_index()
            
            skew_df["skew_improvement"] = skew_df["skew_default"] - skew_df["skew_log10-sqrt"]
            skew_df["skew_rimprovement"] = skew_df["skew_improvement"] / skew_df["skew_default"]
            
            improvement = np.logical_and(skew_df["skew_improvement"] > 0.25, 
                                        skew_df["skew_rimprovement"] > 0.25)
            improvement = np.logical_and(improvement,
                                        skew_df["skew_default"] > 1.5)
            skew_df["improvement"] = improvement
            
            if dataset == "input":
                # Set all initial values to true
                # - except UpperLayerStorage, LowerLayerStorage as they are normally distributed
                # - except TimestepsToAverageDischarge as it is not a quantity
                initial_sel = np.array(["initial" in feature and "LayerStorage" not in feature and "TimestepsToAverageDischarge" not in feature for feature in skew_df["feature"]])
                skew_df.loc[initial_sel, "improvement"] = True
                
                # Set al relative elevation values to true
                dem_improvement = True
                dzRel_sel = np.array(["dzRel" in feature for feature in skew_df["feature"]])
                skew_df.loc[dzRel_sel, "improvement"] = dem_improvement

                # Set all variables available in various sections to their average
                # - example is land-cover input
                skew_df_mean = skew_df.groupby("name")["improvement"].mean()
                skew_df_count = skew_df.groupby("name")["improvement"].count()
                skew_df_agg = {"name": skew_df_mean.index,
                            "mean": skew_df_mean,
                            "count": skew_df_count}
                skew_df_agg = pd.DataFrame(skew_df_agg)
                
                skew_df_agg = skew_df_agg.loc[skew_df_agg["count"] > 1]
                skew_df_agg["improvement"] = skew_df_agg["mean"] > 0.5
                
                agg_names = skew_df_agg.loc[skew_df_agg["improvement"], "name"]
                agg_sel = np.array([name in agg_names for name in skew_df["name"]])
                skew_df.loc[agg_sel, "improvement"] = True
                
                agg_names = skew_df_agg.loc[~skew_df_agg["improvement"], "name"]
                agg_sel = np.array([name in agg_names for name in skew_df["name"]])
                skew_df.loc[agg_sel, "improvement"] = False
            
            skew_out.parent.mkdir(parents=True, exist_ok=True)
            skew_df.to_csv(skew_out)
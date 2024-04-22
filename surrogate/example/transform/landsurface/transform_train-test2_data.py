import pathlib as pl
import pickle
import re

import numpy as np
import pandas as pd

from surrogate.utils.data import load_amt_dataset
from surrogate.nn.functional import StandardTransformer
from surrogate.nn.functional import MinMaxTransformer
from surrogate.nn.functional import LogSqrtStandardTransformer
from surrogate.nn.functional import LogSqrtMinMaxTransformer
from surrogate.nn.functional import PowStandardTransformer
from surrogate.nn.functional import LogStandardTransformer

from surrogate.run.transform.utils.generate_transformer import generate_transformer_array

save_dir = pl.Path("./saves/train-test2")
setup_dir = pl.Path("../../setup/landsurface/saves/train-test2")
out_dir = pl.Path("./saves/train-test2")
trainset = "train"
transformer_types = ["min-max", "standard",
                     "sqrt_min-max", "sqrt_standard",
                     "log_min-max", "log_standard",
                     "log-sqrt_min-max", "log-sqrt_standard",
                     "log10_min-max", "log10_standard",
                     "log10-sqrt_min-max", "log10-sqrt_standard"]
transformer_types += ["pow-0p4_standard",
                     "pow-0p5_standard",
                     "pow-0p6_standard"]
transformer_types += ["losg-1p0_standard",
                      "losg-0p1_standard",
                      "losg-0p01_standard",
                      "losg-0p001_standard"]

resolution = "30min"
    
save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
setup_resolution_dir = pl.Path("{}/{}".format(setup_dir, resolution))
out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))

setup_trainset_dir = pl.Path("{}/{}".format(setup_resolution_dir, trainset))

datasets = [dir.stem for dir in setup_trainset_dir.iterdir() if dir.is_dir()]
# datasets = ["input"]

transformer_type = transformer_types[0]
for transformer_type in transformer_types:
    print("\tTransformer type: {}".format(transformer_type))

    out_type_dir = pl.Path("{}/{}".format(out_resolution_dir, transformer_type))
    
    dataset = datasets[0]
    for dataset in datasets:
        print("\t\tDataset: {}".format(dataset))

        save_dataset_dir = pl.Path("{}/{}".format(save_resolution_dir, dataset))
        setup_dataset_dir = pl.Path("{}/{}".format(setup_trainset_dir, dataset))
        out_dataset_dir = pl.Path("{}/{}".format(out_type_dir, dataset))

        features = np.unique([file.stem for file in setup_dataset_dir.rglob("*.npy") if file.is_file()])
        
        skew_file = pl.Path("{}_skew.csv".format(save_dataset_dir))
        skew_df = pd.read_csv(skew_file, keep_default_na=False).fillna("")
        
        feature = features[0]
        for feature in features:
                            
            out_transformer = pl.Path("{}/{}_transformer.pkl".format(out_dataset_dir, feature))
            if out_transformer.exists():
                print("Already exists")
                continue
                    
            array, meta, _ = load_amt_dataset(feature=feature,
                                                array_dir=setup_dataset_dir,
                                                meta_dir=setup_dataset_dir,
                                                verbose = 0)

            if "min-max" in transformer_type:
                transformer = MinMaxTransformer()
            elif "standard" in transformer_type:
                transformer = StandardTransformer()
            else:
                raise ValueError("Type {} could not be processed".format(transformer_type))
            
            do_logsqrt = skew_df.loc[skew_df["feature"] == feature, "improvement"].values[0]
            if do_logsqrt:
                if "sqrt" in transformer_type or "log" in transformer_type:
                    add_sqrt = "sqrt" in transformer_type
                    add_log = "log" in transformer_type
                    log_10 = "log10" in transformer_type
                    
                    if "min-max" in transformer_type:
                            transformer = LogSqrtMinMaxTransformer(log_10=log_10,
                                                                add_log=add_log,
                                                                add_sqrt=add_sqrt)
                    elif "standard" in transformer_type:
                            transformer = LogSqrtStandardTransformer(log_10 = log_10,
                                                                    add_log=add_log,
                                                                    add_sqrt=add_sqrt)
                    else:
                        raise ValueError("Unkown transformer type")
                    
                if "pow" in transformer_type:
                    power = transformer_type.split("pow-")[1].split("_")[0]
                    power = float(re.sub("p", ".", power))
                    
                    if "standard" in transformer_type:
                            transformer = PowStandardTransformer(power = power)
                    else:
                        raise ValueError("Unkown transformer type")
                    
                if "locg" in transformer_type:
                    base = transformer_type.split("locg-")[1].split("_")[0]
                    base = float(re.sub("p", ".", base))
                    
                    if "standard" in transformer_type:
                            transformer = LogStandardTransformer(base = base)
                    else:
                        raise ValueError("Unkown transformer type")
                    
                if "losg" in transformer_type:
                    small = transformer_type.split("losg-")[1].split("_")[0]
                    small = float(re.sub("p", ".", small))
                    
                    if "standard" in transformer_type:
                            transformer = LogSqrtStandardTransformer(log_10 = False,
                                                                        add_log = True,
                                                                        add_sqrt = False,
                                                                        small = small)
                    else:
                        raise ValueError("Unkown transformer type")
                    
            transformer = generate_transformer_array(array=array,
                                                        transformer=transformer,
                                                        verbose=0)

            out_transformer.parent.mkdir(parents=True, exist_ok=True)
            with open(out_transformer, 'wb') as file:
                pickle.dump(transformer, file)

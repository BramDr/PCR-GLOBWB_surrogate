import pathlib as pl

import pandas as pd
import torch

from surrogate.utils.data import load_batchsets
from surrogate.nn import build_model_from_state_dict
from utils.calculate_sensitivity import calculate_senstivity

save_dir = pl.Path("./saves/global_30min/total")
prepare_dir = pl.Path("../prepare/saves/global_30min")
transform_dir = pl.Path("../transform/saves/global_30min")
upstream_file = pl.Path("../setup/saves/global_30min/upstream.csv")
dir_out = pl.Path("./saves/global_30min/total")
seed = 19920232

upstream = pd.read_csv(upstream_file, index_col=0)

# Load output template
subset = "hyper"
prepare_subset_dir = pl.Path("{}/{}".format(prepare_dir, subset))
transform_subset_dir = pl.Path("{}/{}".format(transform_dir, subset))
train_data_dir = pl.Path("{}/cells_training".format(prepare_subset_dir))
train_dataset = load_batchsets(save_dir=train_data_dir,
                            transformer_dir=transform_subset_dir,
                            upstream=upstream,
                            permute=True,
                            seed=seed,
                            sample_size=10,
                            verbose=1)

sens_file = pl.Path("{}/sensitivity.csv".format(dir_out))
sens_file.parent.mkdir(parents=True, exist_ok=True)

state_file = pl.Path("{}/state_dict.pt".format(save_dir))
state_dict = torch.load(state_file, map_location="cpu")
model = build_model_from_state_dict(state_dict=state_dict,
                                    dropout_rate=0,
                                    cuda=True,
                                    verbose=1)

sensitivity = calculate_senstivity(model = model,
                                    dataset = train_dataset)

sensitivity = sensitivity.groupby(["in_feature", "out_feature", "sensitivity"]).mean()
sensitivity = sensitivity.reset_index()
sensitivity = sensitivity.drop(["index"], axis=1)

sensitivity.to_csv(sens_file)


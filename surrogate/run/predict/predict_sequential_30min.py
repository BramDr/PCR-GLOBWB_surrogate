import pathlib as pl
import pandas as pd

import torch

from surrogate.nn import build_model_from_state_dict

from utils.load_predict_sequential import load_predict_sequential

prepare_dir = pl.Path("../prepare/saves/global_30min/sequential")
transformer_dir = pl.Path("../transform/saves/global_30min")
train_dir = pl.Path("../train/saves/global_30min")
upstream_file = pl.Path("../setup/saves/global_30min/upstream.csv")
dir_out = pl.Path("./saves/global_30min/sequential")
train_subset = "train_8"
seed = 19920223

upstream = pd.read_csv(upstream_file, index_col=0)

transformer_subset_dir = pl.Path("{}/{}".format(transformer_dir, train_subset))
train_subset_dir = pl.Path("{}/{}".format(train_dir, train_subset))

state_file = pl.Path("{}/state_dict.pt".format(train_subset_dir))
state_dict = torch.load(state_file, map_location="cpu")
model = build_model_from_state_dict(state_dict=state_dict,
                                    dropout_rate=0,
                                    cuda=True,
                                    verbose=2)

def sort_fn(file: str):
    number = str(file).split("_")[-1]
    return int(number)

sequences = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]
sequences.sort(key=sort_fn)
    
sequence = sequences[0]
for sequence in sequences:
    print("Working on {}".format(sequence), flush=True)

    sequence_dir = pl.Path("{}/{}".format(prepare_dir, sequence))
    sequence_out = pl.Path("{}/{}".format(dir_out, sequence))
    
    subsets_dirs = [file for file in sequence_dir.iterdir() if file.is_dir()]
    
    load_predict_sequential(save_dir = sequence_dir,
                            transformer_dir=transformer_subset_dir,
                            subset_dirs=subsets_dirs,
                            upstream=upstream,
                            model=model,
                            dir_out=sequence_out,
                            verbose=2)
        

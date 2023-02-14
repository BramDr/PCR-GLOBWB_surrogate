from typing import Optional

import pathlib as pl

import torch

from .SurrogateModel import SurrogateModel
from surrogate.utils.data import load_meta

from .build_model import build_model

def load_model(state_file: pl.Path,
               meta_file: Optional[pl.Path] = None,
               dropout_rate: float = 0,
               try_cuda: bool = False,
               verbose: int = 1) -> SurrogateModel:
        
    state_dict = torch.load(state_file, map_location="cpu")
    
    in_keys = [key for key in state_dict.keys() if "pre_sequence" in key and "weight" in key]
    out_keys = [key for key in state_dict.keys() if "post_sequence" in key and "weight" in key]

    input_size = [state_dict[key].shape[1] for key in in_keys][0]
    output_size = [state_dict[key].shape[0] for key in out_keys][0]
    in_hidden_size = [state_dict[key].shape[0] for key in in_keys]
    out_hidden_size = [state_dict[key].shape[1] for key in out_keys]
    n_lstm = [key for key in state_dict.keys() if "lstm.weight_ih" in key]
    
    in_features = None
    out_features = None
    if meta_file is not None:
        meta = load_meta(file = meta_file)
        in_features=meta["in_features"]
        out_features=meta["out_features"]
    
    model = build_model(input_size=input_size,
                        output_size=output_size,
                        n_lstm=len(n_lstm),
                        in_hidden_size=in_hidden_size,
                        out_hidden_size=out_hidden_size,
                        input_features=in_features,
                        output_features=out_features,
                        dropout_rate=dropout_rate,
                        try_cuda=try_cuda,
                        verbose=verbose)

    model.load_state_dict(state_dict=state_dict)

    return model
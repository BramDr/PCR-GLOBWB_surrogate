from __future__ import annotations
from typing import Optional, Sequence
import pathlib as pl
import warnings
import pickle

import torch
import numpy as np

from surrogate.nn.functional import Transformer

        
class SurrogateModel(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_lstm: int = 1,
                 dropout_rate: float = 0,
                 in_hidden_size: Optional[Sequence[int]] = None,
                 out_hidden_size: Optional[Sequence[int]] = None,
                 input_features: Optional[np.ndarray] = None,
                 output_features: Optional[np.ndarray] = None,
                 input_transformers: Optional[dict[str, Optional[Transformer]]] = None,
                 output_transformers: Optional[dict[str, Optional[Transformer]]] = None,
                 try_cuda: bool = False,
                 gpu: int = 0,
                 seed: int = 19920223) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.input_features = input_features
        if input_features is not None and len(input_features) != input_size:
            raise ValueError("Input features length is not identical to the input size")
        
        self.input_transformers = input_transformers
        if input_transformers is not None and len(input_transformers) != input_size:
            raise ValueError("Input transformers length is not identical to the input size")

        self.output_features = output_features
        if output_features is not None and len(output_features) != output_size:
            raise ValueError("Output features length is not identical to the output size")

        self.output_transformers = output_transformers
        if output_transformers is not None and len(output_transformers) != output_size:
            raise ValueError("Output transformers length is not identical to the output size")
        
        torch.manual_seed(seed=seed)
                        
        # Input linear
        self.pre_sequence = torch.nn.Sequential()
        if in_hidden_size is not None:
            
            pre_layers = []
            for hidden_size in in_hidden_size:
                    
                linear = torch.nn.Linear(in_features = input_size, out_features = hidden_size)
                pre_layers.append(linear)
                
                dropout = torch.nn.Dropout(p = dropout_rate)
                pre_layers.append(dropout)
                
                input_size = hidden_size
            
            pre_layers.append(torch.nn.ReLU())
            self.pre_sequence= torch.nn.Sequential(*pre_layers)
            
        # Output linear
        self.post_sequence = torch.nn.Sequential()
        if out_hidden_size is not None:
            
            post_layers = []
            for hidden_size in reversed(out_hidden_size):
                
                linear = torch.nn.Linear(in_features = hidden_size, out_features = output_size)
                post_layers.insert(0, linear)
                
                dropout = torch.nn.Dropout(p = dropout_rate)
                post_layers.insert(0, dropout)
                    
                output_size = hidden_size
                
            self.post_sequence = torch.nn.Sequential(*post_layers)
            
        # LSTM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lstm = torch.nn.LSTM(input_size = input_size,
                                      hidden_size = output_size,
                                      num_layers = n_lstm,
                                      dropout = dropout_rate)
        
        self.try_cuda = False            
        if try_cuda and torch.cuda.is_available():
            self.try_cuda = True
            self.to("cuda:{}".format(gpu))
    
    def forward(self,
                input: torch.Tensor,
                hidden: Optional[torch.Tensor] = None,
                cell: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        input = self.pre_sequence(input)
        if hidden is not None and cell is not None:
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
        else:
            output, (hidden, cell) = self.lstm(input)
        output = self.post_sequence(output)

        return output, (hidden, cell)

    def __str__(self):
        string = "{}\n{}\n{}".format(self.pre_sequence, self.lstm, self.post_sequence)
        return string
    
    def save(self,
             out_dir: pl.Path):
    
        state_dict = self.state_dict()
        state_dict_file = pl.Path("{}/state_dict.pt".format(out_dir))
        state_dict_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, state_dict_file)
        
        meta = {"input_features": self.input_features,
                "output_features": self.output_features,
                "input_transformers": self.input_transformers,
                "output_transformers": self.output_transformers}
        meta_file = pl.Path("{}/model_meta.pkl".format(out_dir))
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_file, "wb") as f:
            pickle.dump(meta, f)
    
    @staticmethod
    def load(save_dir: pl.Path,
             dropout_rate: float = 0,
             try_cuda: bool = True,
             gpu: int = 0,
             seed: int = 19920223) -> SurrogateModel:
    
        meta_file = pl.Path("{}/model_meta.pkl".format(save_dir))
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
        
        input_features = meta["input_features"]
        output_features = meta["output_features"]
        input_transformers = meta["input_transformers"]
        output_transformers = meta["output_transformers"]
        
        state_dict_file = pl.Path("{}/state_dict.pt".format(save_dir))
        state_dict = torch.load(state_dict_file, map_location="cpu")
        
        in_keys = [key for key in state_dict.keys() if "pre_sequence" in key and "weight" in key]
        out_keys = [key for key in state_dict.keys() if "post_sequence" in key and "weight" in key]
        input_size = [state_dict[key].shape[1] for key in in_keys][0]
        output_size = [state_dict[key].shape[0] for key in out_keys][0]
        n_lstm = len([key for key in state_dict.keys() if "lstm.weight_ih" in key])
        in_hidden_size = [state_dict[key].shape[0] for key in in_keys]
        out_hidden_size = [state_dict[key].shape[1] for key in out_keys]
        
        model = SurrogateModel(input_size=input_size,
                            output_size=output_size,
                            n_lstm=n_lstm,
                            in_hidden_size=in_hidden_size,
                            out_hidden_size=out_hidden_size,
                            input_features=input_features,
                            output_features=output_features,
                            input_transformers=input_transformers,
                            output_transformers=output_transformers,
                            seed=seed,
                            dropout_rate=dropout_rate,
                            try_cuda=try_cuda,
                            gpu=gpu)
        model.load_state_dict(state_dict=state_dict)
        return model
        
    @staticmethod
    def exists(save_dir: pl.Path) -> bool:
        
        meta_file = pl.Path("{}/model_meta.pkl".format(save_dir))
        state_dict_file = pl.Path("{}/state_dict.pt".format(save_dir))
        
        return meta_file.exists() and state_dict_file.exists()
    
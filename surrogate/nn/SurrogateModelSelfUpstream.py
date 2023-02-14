from typing import Optional, Sequence
import time

import torch

from .SurrogateModel import SurrogateModel

class SurrogateModelSelfUpstream(torch.nn.Module):
    def __init__(self,
                 self_input_size: int,
                 self_output_size: int,
                 upstream_input_size: int,
                 max_upstream: int,
                 self_n_lstm: int = 1,
                 upstream_n_lstm: int = 1,
                 self_in_hidden_size: Optional[Sequence[int]] = None,
                 self_out_hidden_size: Optional[Sequence[int]] = None,
                 upstream_in_hidden_size: Optional[Sequence[int]] = None,
                 upstream_out_hidden_size: Optional[Sequence[int]] = None,
                 self_in_features: Optional[Sequence] = None,
                 self_out_features: Optional[Sequence] = None,
                 upstream_in_features: Optional[Sequence] = None,
                 dropout_rate: float = 0,
                 try_cuda: bool = False,
                 seed: int = 19920223) -> None:
        super().__init__()
        
        upstream_output_size = 1
        upstream_out_features = ["upstream_inflow"]
        
        self.upstream_lstm = SurrogateModel(input_size=upstream_input_size,
                                            output_size=upstream_output_size,
                                            n_lstm=upstream_n_lstm,
                                            in_hidden_size=upstream_in_hidden_size,
                                            out_hidden_size=upstream_out_hidden_size,
                                            input_features=upstream_in_features,
                                            output_features=upstream_out_features,
                                            dropout_rate=dropout_rate,
                                            try_cuda=try_cuda,
                                            seed=seed)
        
        #self_input_size += max_upstream
        self_input_size += 1
        if self_in_features is not None and self.upstream_lstm.out_features is not None:
            #upstream_out_features = ["upstream_inflow_{}".format(sample) for sample in range(max_upstream)]
            upstream_out_features = ["upstream_inflow"]
            self_in_features = list(self_in_features) + upstream_out_features
        
        self.self_lstm = SurrogateModel(input_size=self_input_size,
                                        output_size=self_output_size,
                                        n_lstm=self_n_lstm,
                                        in_hidden_size=self_in_hidden_size,
                                        out_hidden_size=self_out_hidden_size,
                                        input_features=self_in_features,
                                        output_features=self_out_features,
                                        dropout_rate=dropout_rate,
                                        try_cuda=try_cuda,
                                        seed=seed)
    
    def forward(self,
                upstream_input: torch.Tensor,
                self_input: torch.Tensor) -> tuple[torch.Tensor, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
        
        n_samples = self_input.shape[0]
        n_upstream_samples = upstream_input.shape[1]
        n_upstream_dates = upstream_input.shape[-2]
        n_upstream_features = upstream_input.shape[-1]
        
        upstream_input = upstream_input.view((-1, n_upstream_dates, n_upstream_features))
        upstream_output, _ = self.upstream_lstm.forward(upstream_input)
        upstream_output = upstream_output.view((n_samples, n_upstream_samples, n_upstream_dates))
        
        upstream_output = torch.sum(input = upstream_output, dim = 1)
        upstream_output = torch.unsqueeze(input = upstream_output, dim = -1)
        self_input = torch.concat((self_input, upstream_output), dim = 2)
        
        output, (hidden, cell) = self.self_lstm.forward(self_input)
        return output, (hidden, cell)

    def __str__(self):
        string = "Upstream {}:\n".format(type(self.upstream_lstm).__name__)
        string += "{}\n".format(self.upstream_lstm)
        string += "Self {}:\n".format(type(self.self_lstm).__name__)
        string += "{}\n".format(self.self_lstm)
        return string

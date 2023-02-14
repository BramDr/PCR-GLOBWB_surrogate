from typing import Optional, Sequence
import warnings

import torch
        
        
class SurrogateModelLinear(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 in_hidden_size: Optional[Sequence[int]] = None,
                 out_hidden_size: Optional[Sequence[int]] = None,
                 input_features: Optional[Sequence] = None,
                 output_features: Optional[Sequence] = None,
                 dropout_rate: float = 0,
                 try_cuda: bool = False,
                 seed: int = 19920223) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.in_features = input_features
        if input_features is None:
            self.in_features = list(range(input_size))
        elif len(input_features) != input_size:
            raise ValueError(
                "Input features length is not identical to the input size")

        self.out_features = output_features
        if output_features is None:
            self.out_features = list(range(output_size))
        elif len(output_features) != output_size:
            raise ValueError(
                "Output features length is not identical to the output size")
        
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
        
        self.try_cuda = False            
        if try_cuda and torch.cuda.is_available():
            self.try_cuda = True
            self.to("cuda")
    
    def forward(self,
                input: torch.Tensor) -> tuple[torch.Tensor, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:

        input = self.pre_sequence(input)
        output = input
        output = self.post_sequence(output)

        return output, (None, None)

    def __str__(self):
        string = "{}\n{}".format(self.pre_sequence, self.post_sequence)
        return string
        
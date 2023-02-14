from typing import Optional, Sequence

import torch

class SurrogateModel(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 n_lstm: int = 1,
                 in_hidden_size: Optional[Sequence[int]] = None,
                 out_hidden_size: Optional[Sequence[int]] = None,
                 in_features: Optional[Sequence] = None,
                 out_features: Optional[Sequence] = None,
                 dropout_rate: float = 0,
                 try_cuda: bool = False,
                 seed: int = 19920223) -> None:
        super().__init__()
        
        self.in_features = in_features
        if in_features is None:
            self.in_features = list(range(input_size))
        elif len(in_features) != input_size:
            raise ValueError(
                "Input features length is not identical to the input size")

        self.out_features = out_features
        if out_features is None:
            self.out_features = list(range(output_size))
        elif len(out_features) != output_size:
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
            
        # LSTM
        self.lstm = torch.nn.LSTM(input_size = input_size,
                                  hidden_size = output_size,
                                  num_layers = n_lstm,
                                  dropout = dropout_rate,
                                  batch_first = True)
        
        self.try_cuda = False            
        if try_cuda and torch.cuda.is_available():
            self.try_cuda = True
            self.to("cuda")
    
    def forward(self,
                input: torch.Tensor,
                hidden: Optional[torch.Tensor] = None,
                cell: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:

        #if hidden is None:
        #    if torch.cuda.is_available():
        #        hidden = torch.zeros(self.lstm.num_layers, input.shape[0], self.lstm.hidden_size, device='cuda:0')
        #    else:
        #        hidden = torch.zeros(self.lstm.num_layers, input.shape[0], self.lstm.hidden_size)
        #if cell is None:
        #    if torch.cuda.is_available():
        #        cell = torch.zeros(self.lstm.num_layers, input.shape[0], self.lstm.hidden_size, device='cuda:0')
        #    else:
        #        cell = torch.zeros(self.lstm.num_layers, input.shape[0], self.lstm.hidden_size)

        input = self.pre_sequence(input)
        output, (hidden, cell) = self.lstm(input)
        output = self.post_sequence(output)

        return output, (hidden, cell)

    def __str__(self):
        string = "{}\n{}\n{}".format(self.pre_sequence, self.lstm, self.post_sequence)
        return string
        
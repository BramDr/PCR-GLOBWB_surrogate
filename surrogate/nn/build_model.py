from typing import Sequence, Optional

from .SurrogateModel import SurrogateModel


def build_model(input_size: int,
                output_size: int,
                n_lstm: int = 1,
                in_hidden_size: Optional[Sequence[int]] = None,
                out_hidden_size: Optional[Sequence[int]] = None,
                input_features: Optional[Sequence] = None,
                output_features: Optional[Sequence] = None,
                dropout_rate: float = 0,
                try_cuda: bool = False,
                seed: int = 19920223,
                verbose: int = 1) -> SurrogateModel:

    model = SurrogateModel(input_size=input_size,
                           output_size=output_size,
                           n_lstm=n_lstm,
                           in_hidden_size=in_hidden_size,
                           out_hidden_size=out_hidden_size,
                           input_features=input_features,
                           output_features=output_features,
                           dropout_rate=dropout_rate,
                           try_cuda=try_cuda,
                           seed=seed)
    model.train(mode=False)

    if verbose > 0:
        print(model, flush=True)

    return model

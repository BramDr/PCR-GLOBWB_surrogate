from typing import Optional, Sequence

import numpy as np
import torch

import torch.utils.data as data

from surrogate.nn.functional import Transformer


class RecurrentDatasetBase(data.Dataset):
    def __init__(self,
                 x: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 samples: Optional[np.ndarray] = None,
                 lons: Optional[np.ndarray] = None,
                 lats: Optional[np.ndarray] = None,
                 dates: Optional[np.ndarray] = None,
                 x_features: Optional[np.ndarray] = None,
                 y_features: Optional[np.ndarray] = None,
                 x_transformers: Optional[dict[str, Optional[Transformer]]] = None,
                 y_transformers: Optional[dict[str, Optional[Transformer]]] = None,) -> None:
        super(RecurrentDatasetBase, self).__init__()

        if x is not None:
            if len(x.shape) < 3:
                raise ValueError(
                    "Input data shape length is not at least 3 (sample_len, sequence_len, feature_len, ...)")

        if y is not None:
            if len(y.shape) < 3:
                raise ValueError(
                    "Output data shape length is not at least 3 (sample_len, sequence_len, feature_len, ...)")

        self.samples_len = 0
        self.dates_len = 0
        self.x_features_len = 0
        self.y_features_len = 0
        if x is not None:
            self.dates_len = x.shape[0]
            self.samples_len = x.shape[-2]
            self.x_features_len = x.shape[-1]
        if y is not None:
            self.dates_len = y.shape[0]
            self.samples_len = y.shape[-2]
            self.y_features_len = y.shape[-1]

        self.samples = np.array([sample for sample in range(self.samples_len)])
        if samples is not None:
            self.samples = samples
        if len(self.samples) != self.samples_len:
            raise ValueError(
                "Samples length ({}) is not identical to the samples dimension ({})".format(len(self.samples),
                                                                                            self.samples_len))

        self.lons = np.array([lon for lon in range(self.samples_len)])
        if lons is not None:
            self.lons = lons
        if len(self.lons) != self.samples_len:
            raise ValueError(
                "Lons length ({}) is not identical to the samples dimension ({})".format(len(self.lons),
                                                                                            self.samples_len))

        self.lats = np.array([lat for lat in range(self.samples_len)])
        if lats is not None:
            self.lats = lats
        if len(self.lats) != self.samples_len:
            raise ValueError(
                "Lats length ({}) is not identical to the samples dimension ({})".format(len(self.lats),
                                                                                            self.samples_len))

        self.dates = np.array([date for date in range(self.dates_len)])
        if dates is not None:
            self.dates = dates
        if len(self.dates) != self.dates_len:
            raise ValueError(
                "Dates length ({}) is not identical to the dates dimension ({})".format(len(self.dates),
                                                                                            self.dates_len))

        self.x_features = np.array([feature for feature in range(self.x_features_len)])
        if x_features is not None:
            self.x_features = x_features
        if len(self.x_features) != self.x_features_len:
            raise ValueError(
                "Input features length ({}) is not identical to the input feature dimension ({})".format(len(self.x_features),
                                                                                            self.x_features_len))

        self.y_features = np.array([feature for feature in range(self.y_features_len)])
        if y_features is not None:
            self.y_features = y_features
        if len(self.y_features) != self.y_features_len:
            raise ValueError(
                "Output features length ({}) is not identical to the output feature dimension ({})".format(len(self.y_features),
                                                                                            self.y_features_len))

        self.x_transformers = dict(zip(self.x_features, [None for _ in range(self.x_features_len)]))
        if x_transformers is not None:
            self.x_transformers = x_transformers
        if len(self.x_transformers) != self.x_features_len:
            raise ValueError(
                "Input features length ({}) is not identical to the input feature dimension ({})".format(len(self.x_transformers),
                                                                                            self.x_features_len))

        self.y_transformers = dict(zip(self.y_features, [None for _ in range(self.y_features_len)]))
        if y_transformers is not None:
            self.y_transformers = y_transformers
        if len(self.y_transformers) != self.y_features_len:
            raise ValueError(
                "Output features length ({}) is not identical to the output feature dimension ({})".format(len(self.y_transformers),
                                                                                            self.y_features_len))

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self,
                    index) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
            
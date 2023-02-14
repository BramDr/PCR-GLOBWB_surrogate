from typing import Optional, Sequence

import numpy as np
import torch

import torch.utils.data as data


class SequenceBaseset(data.Dataset):
    def __init__(self,
                 x: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 samples: Optional[Sequence] = None,
                 lons: Optional[Sequence] = None,
                 lats: Optional[Sequence] = None,
                 dates: Optional[Sequence] = None,
                 x_features: Optional[Sequence] = None,
                 y_features: Optional[Sequence] = None) -> None:
        super(SequenceBaseset, self).__init__()

        if x is not None:
            if len(x.shape) < 3:
                raise ValueError(
                    "Input data shape length is not at least 3 (sample_len, sequence_len, feature_len, ...)")

        if y is not None:
            if len(y.shape) < 3:
                raise ValueError(
                    "Output data shape length is not at least 3 (sample_len, sequence_len, feature_len, ...)")
                
        #if x is not None and y is not None:
        #    if x.shape[:-2] != y.shape[:-2]:
        #        raise ValueError(
        #            "Input ({}) and output ({}) shapes are not identical".format(x.shape,
        #                                                                     y.shape))

        self.samples_len = 0
        self.dates_len = 0
        self.x_features_len = 0
        self.y_features_len = 0
        if x is not None:
            self.samples_len = x.shape[0]
            self.dates_len = x.shape[-2]
            self.x_features_len = x.shape[-1]
        if y is not None:
            self.samples_len = y.shape[0]
            self.dates_len = y.shape[-2]
            self.y_features_len = y.shape[-1]

        self.samples = list(range(self.samples_len))
        if samples is not None:
            self.samples = samples
        if len(self.samples) != self.samples_len:
            raise ValueError(
                "Samples length ({}) is not identical to the samples dimension ({})".format(len(self.samples),
                                                                                            self.samples_len))

        self.lons = list(range(self.samples_len))
        if lons is not None:
            self.lons = lons
        if len(self.lons) != self.samples_len:
            raise ValueError(
                "Lons length ({}) is not identical to the samples dimension ({})".format(len(self.lons),
                                                                                            self.samples_len))

        self.lats = list(range(self.samples_len))
        if lats is not None:
            self.lats = lats
        if len(self.lats) != self.samples_len:
            raise ValueError(
                "Lats length ({}) is not identical to the samples dimension ({})".format(len(self.lats),
                                                                                            self.samples_len))

        self.dates = list(range(self.dates_len))
        if dates is not None:
            self.dates = dates
        if len(self.dates) != self.dates_len:
            raise ValueError(
                "Dates length ({}) is not identical to the dates dimension ({})".format(len(self.dates),
                                                                                            self.dates_len))

        self.x_features = list(range(self.x_features_len))
        if x_features is not None:
            self.x_features = x_features
        if len(self.x_features) != self.x_features_len:
            raise ValueError(
                "Input features length ({}) is not identical to the input feature dimension ({})".format(len(self.x_features),
                                                                                            self.x_features_len))

        self.y_features = list(range(self.y_features_len))
        if y_features is not None:
            self.y_features = y_features
        if len(self.y_features) != self.y_features_len:
            raise ValueError(
                "Output features length ({}) is not identical to the output feature dimension ({})".format(len(self.y_features),
                                                                                            self.y_features_len))

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self,
                    index) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
            
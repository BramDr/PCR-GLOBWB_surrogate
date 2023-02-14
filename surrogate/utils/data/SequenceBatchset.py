from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .SequenceBaseset import SequenceBaseset


class SequenceBatchset(SequenceBaseset):
    def __init__(self,
                 x: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 samples_size: Optional[int] = None,
                 dates_size: Optional[int] = None,
                 samples: Optional[Sequence] = None,
                 lons: Optional[Sequence] = None,
                 lats: Optional[Sequence] = None,
                 dates: Optional[Sequence] = None,
                 x_features: Optional[Sequence] = None,
                 y_features: Optional[Sequence] = None,
                 squeeze_batch: bool = False,
                 drop_last: bool = False,
                 cuda: bool = False) -> None:
        super(SequenceBatchset, self).__init__(x=x,
                                               y=y,
                                               samples=samples,
                                               lons=lons,
                                               lats=lats,
                                               dates=dates,
                                               x_features=x_features,
                                               y_features=y_features)
        
        self.drop_last = drop_last

        self.samples_size = self.samples_len
        self.dates_size = self.dates_len
        if samples_size is not None:
            self.samples_size = samples_size
        if dates_size is not None:
            self.dates_size = dates_size

        self.x_items = []
        self.y_items = []
        self.sample_mapping = []
        self.date_mapping = []
        for sample_index, sample_start in enumerate(range(0, self.samples_len, self.samples_size)):
            sample_end = sample_start + self.samples_size

            if sample_end > self.samples_len and drop_last:
                continue

            for date_index, date_start in enumerate(range(0, self.dates_len, self.dates_size)):
                date_end = date_start + self.dates_size

                if date_end > self.dates_len and drop_last:
                    continue
                
                batch_x = None
                if x is not None:
                    batch_x = x
                    batch_x = batch_x[sample_start:sample_end, ...]
                    batch_x = batch_x[..., date_start:date_end, :]
                    batch_x = np.copy(batch_x)
                    if squeeze_batch:
                        batch_x = np.squeeze(batch_x)
                    batch_x = torch.from_numpy(batch_x)
                
                batch_y = None
                if y is not None:
                    batch_y = y
                    batch_y = batch_y[sample_start:sample_end, ...]
                    batch_y = batch_y[..., date_start:date_end, :]
                    batch_y = np.copy(batch_y)
                    if squeeze_batch:
                        batch_y = np.squeeze(batch_y)
                    batch_y = torch.from_numpy(batch_y)

                self.x_items.append(batch_x)
                self.y_items.append(batch_y)
                self.sample_mapping.append(sample_index)
                self.date_mapping.append(date_index)

        self.sample_mapping = np.array(self.sample_mapping)
        self.date_mapping = np.array(self.date_mapping)
        self.batch_len = len(self.x_items)
        
        self.cuda = cuda
        if self.cuda and torch.cuda.is_available():
            self.x_items = [item.cuda() for item in self.x_items if item is not None]
            self.y_items = [item.cuda() for item in self.y_items if item is not None]
        
    def __str__(self):
        x, y = self[0]
        y_shape = None
        x_shape = None
        if x is not None:
            x_shape = x.shape
        if y is not None:
            y_shape = y.shape
            
        string = "{}: {} ({}, {})".format(type(self).__name__,
                                         self.batch_len,
                                            x_shape,
                                            y_shape)
        return string

    def __len__(self):
        return self.batch_len

    def __getitem__(self,
                    index) -> tuple[torch.Tensor, torch.Tensor]:
        x_item = self.x_items[index]
        y_item = self.y_items[index]
        
        return x_item, y_item

    def _get_x_y(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = None
        if self.x_items[0] is not None:
            x = self._reconstruct(self.x_items)

        y = None
        if self.y_items[0] is not None:
            y = self._reconstruct(self.y_items)

        return x, y

    def _reconstruct(self,
                     tensors: list[torch.Tensor]) -> torch.Tensor:

        sample_tensors = []
        for sample_index in pd.unique(self.sample_mapping):
            indices = np.where(self.sample_mapping == sample_index)[0]
            date_tensors = [tensors[index] for index in indices]

            sample_tensor = torch.concat(date_tensors, dim=1)
            sample_tensors.append(sample_tensor)

        tensor = torch.concat(sample_tensors, dim=0)

        return tensor

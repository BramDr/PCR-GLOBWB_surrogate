from typing import Union, Sequence, Optional
import torch.utils.data as data

from .SequenceBatchset import SequenceBatchset
from .SequenceDataLoader import SequenceDataLoader
from .SequencePreFetcher import SequencePreFetcher

def load_dataloader(dataset: SequenceBatchset,
                    sampler: Optional[data.Sampler] = None,
                    allow_prefetcher: bool = False,
                     verbose: int = 1) -> Union[SequenceDataLoader, SequencePreFetcher]:
        
    dataloader = SequenceDataLoader(dataset=dataset,
                                    sampler=sampler)
    
    dataset_item = dataset[0][0]
    batchset_device = dataset_item.device.type
            
    if batchset_device != "cuda" and allow_prefetcher:
        dataloader = SequencePreFetcher(dataloader=dataloader)
    
    if verbose > 0:            
        print("Loaded {}".format(dataloader), flush=True)
    
    return dataloader


def load_dataloaders(datasets: Sequence[SequenceBatchset],
                     samplers: Optional[Sequence[data.Sampler]] = None,
                     allow_prefetcher: bool = False,
                     verbose: int = 1) -> Sequence[Union[SequenceDataLoader, SequencePreFetcher]]:
    
    if samplers is None:
        samplers = [None] * len(datasets)
    
    dataloaders = []
    for dataset, sampler in zip(datasets, samplers):
        dataloader = load_dataloader(dataset=dataset,
                                     sampler=sampler,
                                     allow_prefetcher=allow_prefetcher,
                                     verbose=verbose)
        dataloaders.append(dataloader)
        
    return dataloaders
    
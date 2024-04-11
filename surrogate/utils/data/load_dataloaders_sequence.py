from typing import Union, Sequence, Optional
import torch.utils.data as data

from .RecurrentBatchsetSequence import RecurrentBatchsetSequence
from .DataLoaderSequence import DataLoaderSequence
from .PreFetcherSequence import PreFetcherSequence

def load_dataloader_sequence(dataset: RecurrentBatchsetSequence,
                             sampler: Optional[data.Sampler] = None,
                             allow_prefetcher: bool = False,
                             verbose: int = 1) -> Union[DataLoaderSequence, PreFetcherSequence]:
        
    dataloader = DataLoaderSequence(dataset=dataset,
                                    sampler=sampler)
    
    dataset_item = dataset[0][0][0]
    batchset_device = dataset_item.device.type
            
    if batchset_device != "cuda" and allow_prefetcher:
        dataloader = PreFetcherSequence(dataloader=dataloader)
    
    if verbose > 0:            
        print("Loaded {}".format(dataloader), flush=True)
    
    return dataloader


def load_dataloaders_sequence(datasets: Sequence[RecurrentBatchsetSequence],
                              samplers: Optional[Sequence[Optional[data.Sampler]]] = None,
                              allow_prefetcher: bool = False,
                              verbose: int = 1) -> list[Union[DataLoaderSequence, PreFetcherSequence]]:
    
    if samplers is None:
        samplers = [None for _ in range(len(datasets))]
    
    dataloaders = []
    for dataset, sampler in zip(datasets, samplers):
        dataloader = load_dataloader_sequence(dataset=dataset,
                                     sampler=sampler,
                                     allow_prefetcher=allow_prefetcher,
                                     verbose=verbose)
        dataloaders.append(dataloader)
        
    return dataloaders
    
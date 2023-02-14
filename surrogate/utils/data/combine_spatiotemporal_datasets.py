from .SequenceBaseset import SequenceBaseset
from .concatenate_datasets import concatenate_datasets


def combine_spatiotemporal_datasets(train: SequenceBaseset,
                                    spatial_test: SequenceBaseset,
                                    temporal_test: SequenceBaseset,
                                    spatiotemporal_test: SequenceBaseset,
                                    verbose: int = 1) -> SequenceBaseset:

    spatial = concatenate_datasets(datasets=[train, spatial_test],
                                   direction="sample",
                                   verbose=verbose-1)

    temporal = concatenate_datasets(datasets=[temporal_test, spatiotemporal_test],
                                    direction="sample",
                                    verbose=verbose-1)

    dataset = concatenate_datasets(datasets=[spatial, temporal],
                                   direction="date",
                                   verbose=verbose-1)

    if verbose > 0:
        print("Combined train and test datasets", flush=True)

    return dataset

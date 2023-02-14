import numpy as np

from .SequenceBaseset import SequenceBaseset
from .SequenceBatchset import SequenceBatchset


def concatenate_datasets(datasets: list[SequenceBaseset],
                         direction: str = "spatial",
                         verbose: int = 1) -> SequenceBatchset:

    if direction == "sample":
        concatenate_axis = 0
    elif direction == "date":
        concatenate_axis = -2
    elif direction == "feature":
        concatenate_axis = -1
    else:
        raise NotImplementedError("Cannot concatinate array direction")

    xs = []
    ys = []
    for dataset in datasets:
        x, y = dataset._get_x_y()
        xs.append(x.cpu().numpy())
        if y is not None:
            ys.append(y.cpu().numpy())

    x = np.concatenate(xs, axis=concatenate_axis)
    y = None
    if len(ys) > 0:
        y = np.concatenate(ys, axis=concatenate_axis)

    samples = datasets[0].samples
    lats = datasets[0].lats
    lons = datasets[0].lons
    dates = datasets[0].dates
    in_features = datasets[0].x_features
    out_features = datasets[0].y_features
    
    if direction == "sample":
        samples = [sample for dataset in datasets if dataset.samples is not None for sample in dataset.samples]
        lons = [lon for dataset in datasets if dataset.lons is not None for lon in dataset.lons]
        lats = [lat for dataset in datasets if dataset.lats is not None for lat in dataset.lats]
    elif direction == "date":
        dates = [datum for dataset in datasets if dataset.dates is not None for datum in dataset.dates]
    elif direction == "feature":
        in_features = [
            feature for dataset in datasets if dataset.x_features is not None for feature in dataset.x_features]
        out_features = [
            feature for dataset in datasets if dataset.y_features is not None for feature in dataset.y_features]

    sample_size = datasets[0].samples_size
    drop_last = datasets[0].drop_last
    cuda = datasets[0].cuda
    
    concatenated_datasets = SequenceBatchset(x=x,
                                             y=y,
                                             samples_size=sample_size,
                                             samples=samples,
                                             lats=lats,
                                             lons=lons,
                                             dates=dates,
                                             x_features=in_features,
                                             y_features=out_features,
                                             drop_last=drop_last,
                                             cuda=cuda)

    if verbose > 0:
        x, y = concatenated_datasets[0]
        if y is None:
            print("Concatenated {}: {} ({} None)".format(type(concatenated_datasets).__name__,
                                                    len(concatenated_datasets),
                                                    x.shape), flush=True)
        else:
            print("Concatenated {}: {} ({} {})".format(type(concatenated_datasets).__name__,
                                                    len(concatenated_datasets),
                                                    x.shape,
                                                    y.shape), flush=True)

    return concatenated_datasets

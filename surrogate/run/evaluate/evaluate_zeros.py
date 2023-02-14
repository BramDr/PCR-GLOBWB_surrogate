import pathlib as pl
import numpy as np

from surrogate.utils.data import load_train_test_datasets

save_dir = pl.Path("../preprocess/saves/global_30min/transformed/minmax/cells_training_test")
test_dirs = [pl.Path("../preprocess/saves/global_30min/transformed/minmax/cells_validation_spatial_test"),
             pl.Path("../preprocess/saves/global_30min/transformed/minmax/cells_validation_temporal_test"),
             pl.Path("../preprocess/saves/global_30min/transformed/minmax/cells_validation_spatiotemporal_test")]
seed = 19920232

def zero_fraction(input: np.ndarray):
    n_zeros = np.sum(input == 0)
    n_values = len(input)
    return n_zeros / n_values

dataset, test_datasets, _ = load_train_test_datasets(train_dir=save_dir,
                                                     test_dirs=test_dirs,
                                                     cuda=False)

output = dataset.y
features = dataset.out_features
output_flat = np.reshape(output, (-1,output.shape[2]))
output_zeros = np.apply_along_axis(zero_fraction, axis=0, arr=output_flat)

dict_zeros = {f: v for f,v in zip(features, output_zeros)}

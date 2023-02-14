import pathlib as pl

import pandas as pd
import torch

from surrogate.utils.data import load_batchsets
from surrogate.utils.data import load_batchsets
from surrogate.utils.data import combine_spatiotemporal_datasets
from surrogate.nn import build_model_from_state_dict
from surrogate.utils.plot import array_to_ts_dataframe
from surrogate.utils.plot import array_to_cdf_dataframe
from surrogate.utils.plot import array_to_sp_dataframe

save_dir = pl.Path("./saves/global_30min")
prepare_dir = pl.Path("../prepare/saves/global_30min")
transform_dir = pl.Path("../transform/saves/global_30min")
upstream_file = pl.Path("../setup/saves/global_30min/upstream.csv")
dir_out = pl.Path("./saves/global_30min")
seed = 19920232

upstream = pd.read_csv(upstream_file, index_col=0)

# Load output template
subset = "train_96"
prepare_subset_dir = pl.Path("{}/{}".format(prepare_dir, subset))
transform_subset_dir = pl.Path("{}/{}".format(transform_dir, subset))
train_data_dir = pl.Path("{}/cells_training".format(prepare_subset_dir))
train_dataset = load_batchsets(save_dir=train_data_dir,
                            transformer_dir=transform_subset_dir,
                            upstream=upstream,
                            permute=True,
                            seed=seed,
                            sample_size=10,
                            verbose=1)
out_features = train_dataset.y_features
del train_dataset

subsets = ["train_{}".format(size)
           for size in [8, 16, 32, 48, 64, 80, 96]]
subsets.reverse()

subset = "train_96"
for subset in subsets:
    print("Working on {}".format(subset), flush = True)

    save_subset_dir = pl.Path("{}/{}".format(save_dir, subset))
    prepare_subset_dir = pl.Path("{}/{}".format(prepare_dir, subset))
    transform_subset_dir = pl.Path("{}/{}".format(transform_dir, subset))
    subset_out = pl.Path("{}/{}".format(dir_out, subset))

    predicted_file = pl.Path("{}/array_predicted.npy".format(subset_out))
    predicted_file.parent.mkdir(parents=True, exist_ok=True)
    ts_file = pl.Path("{}/output_ts.csv".format(subset_out))
    ts_file.parent.mkdir(parents=True, exist_ok=True)
    cdf_file = pl.Path("{}/output_cdf.csv".format(subset_out))
    cdf_file.parent.mkdir(parents=True, exist_ok=True)
    sp_file = pl.Path("{}/output_sp.csv".format(subset_out))
    sp_file.parent.mkdir(parents=True, exist_ok=True)

    train_data_dir = pl.Path("{}/cells_training".format(prepare_subset_dir))
    test_data_dirs = [pl.Path("{}/cells_validation_spatial".format(prepare_subset_dir)),
                    pl.Path("{}/cells_validation_temporal".format(prepare_subset_dir)),
                    pl.Path("{}/cells_validation_spatiotemporal".format(prepare_subset_dir))]
    
    train_dataset = load_batchsets(save_dir=train_data_dir,
                                transformer_dir=transform_subset_dir,
                                include_output=False,
                                upstream=upstream,
                                permute=True,
                                seed=seed,
                                sample_size=10,
                                verbose=1)
    
    test_datasets = load_batchsets(save_dirs=test_data_dirs,
                                transformer_dir=transform_subset_dir,
                                include_output=False,
                                upstream=upstream,
                                permute=True,
                                seed=seed,
                                sample_size=10,
                                verbose=2)
    
    dataset = combine_spatiotemporal_datasets(train=train_dataset,
                                              spatial_test=test_datasets[0],
                                              temporal_test=test_datasets[1],
                                              spatiotemporal_test=test_datasets[2],
                                              verbose=2)
    del train_dataset
    del test_datasets
    
    state_file = pl.Path("{}/state_dict.pt".format(save_subset_dir))
    state_dict = torch.load(state_file, map_location="cpu")
    model = build_model_from_state_dict(state_dict=state_dict,
                                        dropout_rate=0,
                                        cuda=True,
                                        verbose=1)
    
    y_preds = []
    for index in range(len(dataset)):
        x, _ = dataset[index]
        with torch.inference_mode():
            y_pred, _ = model.forward(x.cuda())
            y_pred = y_pred.detach()
        y_preds.append(y_pred)
    y_pred = torch.concat(y_preds, dim = 0).cpu().numpy()

    #np.save(file = predicted_file, arr = y_pred)
    
    plot_df = array_to_ts_dataframe(input=y_pred,
                                    features=out_features,
                                    sequences=dataset.dates,
                                    save_sensitivity=True,
                                    save_range=True)
    plot_df.to_csv(ts_file)

    plot_df = array_to_cdf_dataframe(input=y_pred,
                                    features=out_features,
                                    max_values=1000)
    plot_df.to_csv(cdf_file)
    
    if dataset.lats is None or dataset.lons is None:
        raise ValueError("No lats or lons found in dataset")
    
    plot_df = array_to_sp_dataframe(input=y_pred,
                                    features=out_features,
                                    lats=dataset.lats,
                                    lons=dataset.lons,
                                    save_variability=True,
                                    save_range=True)
    plot_df.to_csv(sp_file)

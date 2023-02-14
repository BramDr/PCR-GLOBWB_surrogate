import pathlib as pl
import pandas as pd

import torch.utils.data as data
import torch.optim as optim

from surrogate.utils.data import load_batchset
from surrogate.utils.data import load_batchsets
from surrogate.utils.data import load_dataloader
from surrogate.utils.data import load_dataloaders
from surrogate.nn import SurrogateModel
from surrogate.utils.train import PerformanceLogger
from surrogate.utils.train import BestModelLogger
from surrogate.utils.train import ModuleTrainer
from surrogate.nn.functional import SequenceMseMetric

feature_dir = pl.Path("../features/saves/global_05min")
prepare_dir = pl.Path("../prepare/saves/global_05min")
transformer_dir = pl.Path("../transform/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
seed = 19920232
samples_size = 32
dates_size = 365
hidden_size = 128
dropout_rate = 0.5
learning_rate = 1e-4
n_backwards = 5e4

in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
in_features = pd.read_csv(in_features_file, keep_default_na=False).fillna("")
out_features = pd.read_csv(out_features_file, keep_default_na=False).fillna("")

submasks = [dir.stem for dir in prepare_dir.iterdir() if dir.is_dir()]
submasks = ["cells_M17"]

submask = "cells_M17"
#submask = submasks[0]
#for submask in submasks:
#    print("Working on {}".format(submask))
    
prepare_submask_dir = pl.Path("{}/{}".format(prepare_dir, submask))
transformer_submask_dir = pl.Path("{}/{}".format(transformer_dir, submask))
submask_out = pl.Path("{}/{}".format(dir_out, submask))

trainsets = [dir.stem for dir in prepare_submask_dir.iterdir() if dir.is_dir() if "train_" in dir.stem or "hyper" in dir.stem]

trainset = "train_32"
for trainset in trainsets:
    print("Working on {}".format(trainset), flush = True)

    prepare_trainset_dir = pl.Path("{}/{}".format(prepare_submask_dir, trainset))
    transformer_trainset_dir = pl.Path("{}/{}".format(transformer_submask_dir, trainset))
    trainset_out = pl.Path("{}/{}".format(submask_out, trainset))

    train_dir = pl.Path("{}/cells_training".format(prepare_trainset_dir))
    test_dirs = [pl.Path("{}/cells_validation_spatial".format(prepare_trainset_dir)),
                    pl.Path("{}/cells_validation_temporal".format(prepare_trainset_dir)),
                    pl.Path("{}/cells_validation_spatiotemporal".format(prepare_trainset_dir))]

    train_dataset = load_batchset(save_dir=train_dir,
                                    input_features=in_features["feature"].to_list(),
                                    output_features=out_features["feature"].to_list(),
                                    transformer_dir=transformer_trainset_dir,
                                    sample_size=samples_size,
                                    dates_size=dates_size,
                                    verbose=2)

    #train_sampler = data.RandomSampler(data_source=train_dataset)
    train_dataloader = load_dataloader(dataset=train_dataset,
                                    #sampler=train_sampler,
                                    allow_prefetcher=True)

    test_datasets = load_batchsets(save_dirs=test_dirs,
                                    input_features=in_features["feature"].to_list(),
                                    output_features=out_features["feature"].to_list(),
                                    transformer_dir=transformer_trainset_dir,
                                    sample_size=samples_size,
                                    verbose=2)
        
    test_dataloaders = load_dataloaders(datasets=test_datasets,
                                        allow_prefetcher=True)

    # Model
    model = SurrogateModel(input_size=train_dataset.x_features_len,
                            output_size=train_dataset.y_features_len,
                            in_hidden_size=[hidden_size],
                            out_hidden_size=[hidden_size],
                            input_features=train_dataset.x_features,
                            output_features=train_dataset.y_features,
                            dropout_rate=dropout_rate,
                            try_cuda=True,
                            seed=seed)
    model.train(mode=False)
    print(model, flush=True)

    # Trainer
    callbacks = []
    callback_out = pl.Path("{}/statistics.csv".format(trainset_out))
    callback = PerformanceLogger(path=callback_out, delayed=False)
    callbacks.append(callback)
    callback_out = pl.Path("{}/state_dict.pt".format(trainset_out))
    callback = BestModelLogger(path=callback_out, delayed=False, verbose=True)
    callbacks.append(callback)

    optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate)
    trainer = ModuleTrainer(train_loader=train_dataloader,
                            model=model,
                            optimizer=optimizer,
                            callbacks=callbacks,
                            test_loaders=test_dataloaders)
    print(trainer, flush=True)

    # Train!
    epochs = int(n_backwards / len(train_dataset))
    print("Runnning {} epochs".format(epochs), flush=True)
    best_loss = trainer.run(epochs=epochs,
                            loss = SequenceMseMetric(),
                            seed=seed)
    print(best_loss, flush=True)

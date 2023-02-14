import pathlib as pl

import pandas as pd
import torch
import torch.utils.data as data
import torch.optim as optim

from surrogate.utils.data import load_batchset
from surrogate.utils.data import load_batchsets
from surrogate.utils.data import SequenceBatchsetSelfUpstream
from surrogate.utils.data import load_dataloader
from surrogate.utils.data import load_dataloaders
from surrogate.utils.data import SequencePreFetcherSelfUpstream
from surrogate.nn import SurrogateModelSelfUpstream
from surrogate.utils.train import PerformanceLogger
from surrogate.utils.train import BestModelLogger
from surrogate.utils.train import ModuleTrainerSelfUpstream
from surrogate.nn.functional import SequenceMseMetric

feature_dir = pl.Path("../features/saves/global_30min")
prepare_dir = pl.Path("../prepare/saves/global_30min")
transformer_dir = pl.Path("../transform/saves/global_30min")
dir_out = pl.Path("./saves/global_30min/total")
seed = 19920232
samples_size = 32
dates_size = 365
n_backwards = 5e5
subset = "train_8"
subset = "hyper" #TODO: remove

upstream_file = pl.Path("{}/upstream_cd.csv".format(feature_dir))
upstream = pd.read_csv(upstream_file, index_col=0)

self_in_features_file = pl.Path("{}/features_self_input.csv".format(feature_dir))
self_out_features_file = pl.Path("{}/features_self_output.csv".format(feature_dir))
upstream_in_features_file = pl.Path("{}/features_upstream_input.csv".format(feature_dir))
upstream_out_features_file = pl.Path("{}/features_upstream_output.csv".format(feature_dir))
self_in_features = pd.read_csv(self_in_features_file, keep_default_na=False).fillna("")
self_out_features = pd.read_csv(self_out_features_file, keep_default_na=False).fillna("")
upstream_in_features = pd.read_csv(upstream_in_features_file, keep_default_na=False).fillna("")
upstream_out_features = pd.read_csv(upstream_out_features_file, keep_default_na=False).fillna("")

prepare_subset_dir = pl.Path("{}/{}".format(prepare_dir, subset))
transformer_subset_dir = pl.Path("{}/{}".format(transformer_dir, subset))

train_dir = pl.Path("{}/cells_training".format(prepare_subset_dir))
test_dirs = [pl.Path("{}/cells_validation_spatial".format(prepare_subset_dir)),
             pl.Path("{}/cells_validation_temporal".format(prepare_subset_dir)),
             pl.Path("{}/cells_validation_spatiotemporal".format(prepare_subset_dir))]

## Training data and dataloaders
train_self_dataset = load_batchset(save_dir=train_dir,
                                   input_features=self_in_features["feature"].to_list(),
                                   output_features=self_out_features["feature"].to_list(),
                                   transformer_dir=transformer_subset_dir,
                                   sample_size=samples_size,
                                   dates_size=dates_size)
train_upstream_dataset = load_batchset(save_dir=train_dir,
                                       input_features=upstream_in_features["feature"].to_list(),
                                       output_features=upstream_out_features["feature"].to_list(),
                                       transformer_dir=transformer_subset_dir,
                                       sample_size=samples_size,
                                       dates_size=dates_size,
                                       upstream=upstream,
                                       permute=True,
                                       seed=seed)
train_dataset = SequenceBatchsetSelfUpstream(self_batchset = train_self_dataset,
                                             upstream_batchset = train_upstream_dataset)
print("Loaded {}".format(train_dataset), flush=True)

print(torch.cuda.memory_summary())

train_self_generator = torch.Generator().manual_seed(seed)
train_self_sampler = data.RandomSampler(data_source=train_self_dataset,
                                   generator=train_self_generator)
train_self_dataloader = load_dataloader(dataset=train_self_dataset,
                                        sampler=train_self_sampler,
                                        allow_prefetcher=True)
train_upstream_generator = torch.Generator().manual_seed(seed)
train_upstream_sampler = data.RandomSampler(data_source=train_self_dataset,
                                   generator=train_upstream_generator)
train_upstream_dataloader = load_dataloader(dataset=train_upstream_dataset,
                                            sampler=train_upstream_sampler,
                                            allow_prefetcher=True)
train_dataloader = SequencePreFetcherSelfUpstream(self_dataloader = train_self_dataloader,
                                                  upstream_dataloader= train_upstream_dataloader)
print("Loaded {}".format(train_dataloader), flush=True)

print(torch.cuda.memory_summary())

## Testing data and dataloaders
test_self_datasets= []
test_upstream_datasets = []

#test_self_datasets = load_batchsets(save_dirs=test_dirs,
#                                    transformer_dir=transformer_subset_dir,
#                                    input_features=self_in_features["feature"].to_list(),
#                                    output_features=self_out_features["feature"].to_list(),
#                                    sample_size=samples_size,
#                                    verbose=2)

#test_upstream_datasets = load_batchsets(save_dirs=test_dirs,
#                                        transformer_dir=transformer_subset_dir,
#                                        input_features=upstream_in_features["feature"].to_list(),
#                                        output_features=upstream_out_features["feature"].to_list(),
#                                        sample_size=samples_size,
#                                        upstream=upstream,
#                                        permute=True,
#                                        seed=seed,
#                                        verbose=2)

test_datasets = []
for test_self_dataset, test_upstream_dataset in zip(test_self_datasets, test_upstream_datasets):
    test_dataset = SequenceBatchsetSelfUpstream(self_batchset = test_self_dataset,
                                                upstream_batchset = test_upstream_dataset)
    print("Loaded {}".format(test_dataset), flush=True)
    test_datasets.append(test_dataset)
    
test_self_dataloaders = load_dataloaders(datasets=test_self_datasets,
                                         allow_prefetcher=True)
test_upstream_dataloaders = load_dataloaders(datasets=test_upstream_datasets,
                                             allow_prefetcher=True)
test_dataloaders = []
for test_self_dataloader, test_upstream_dataloader in zip(test_self_dataloaders, test_upstream_dataloaders):
    test_dataloader = SequencePreFetcherSelfUpstream(self_dataloader = test_self_dataloader,
                                                  upstream_dataloader = test_upstream_dataloader)
    print("Loaded {}".format(test_dataloader), flush=True)
    test_dataloaders.append(test_dataloader)

# Model
max_upstream = train_upstream_dataset[0][0].shape[1]
model = SurrogateModelSelfUpstream(self_input_size=train_dataset.self_batchset.x_features_len,
                                   self_output_size=train_dataset.self_batchset.y_features_len,
                                   upstream_input_size=train_dataset.upstream_batchset.x_features_len,
                                   max_upstream=max_upstream,
                                   self_n_lstm=1,
                                   upstream_n_lstm=1,
                                   self_in_hidden_size=[256],
                                   self_out_hidden_size=[256],
                                   upstream_in_hidden_size=[32],
                                   upstream_out_hidden_size=[32],
                                   self_in_features=train_dataset.self_batchset.x_features,
                                   self_out_features=train_dataset.self_batchset.y_features,
                                   upstream_in_features=train_dataset.upstream_batchset.x_features,
                                   dropout_rate=0.5,
                                   try_cuda=True,
                                   seed=seed)
model.train(mode=False)
print(model, flush=True)

print(torch.cuda.memory_summary())

model.train(mode=True)

for (upstream_x, _), (self_x, y_true) in train_dataloader:
    print(torch.cuda.memory_summary())
    
    y_pred, _ = model.forward(upstream_input=upstream_x,
                           self_input=self_x)
    print(torch.cuda.memory_summary())
    y_pred.mean().backward()
    print(torch.cuda.memory_summary())
    break

model.train(mode=False)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

callbacks = []
#callback_out = pl.Path("{}/statistics.csv".format(dir_out))
#callback = PerformanceLogger(path=callback_out, delayed=True)
#callbacks.append(callback)
#callback_out = pl.Path("{}/state_dict.pt".format(dir_out))
#callback = BestModelLogger(path=callback_out, delayed=True)
#callbacks.append(callback)

optimizer = optim.AdamW(params=model.parameters(), lr=5e-4)
trainer = ModuleTrainerSelfUpstream(train_loader=train_dataloader,
                                    model=model,
                                    optimizer=optimizer,
                                    callbacks=callbacks,
                                    test_loaders=test_dataloaders)
print(trainer, flush=True)

epochs = int(n_backwards / len(train_dataset))
print("Runnning {} epochs".format(epochs), flush=True)
best_loss = trainer.run(epochs=epochs,
                        loss = SequenceMseMetric(),
                        seed=seed)
print(best_loss, flush=True)

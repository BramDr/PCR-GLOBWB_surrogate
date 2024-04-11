import pathlib as pl

import pandas as pd

save_dir = pl.Path("./saves/train-test2")
out_dir = pl.Path("./saves/train-test2/mulres")
seed = 19920223

resolutions = [dir.stem for dir in save_dir.iterdir() if dir.is_dir() and dir.stem != "mulres"]
save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolutions[0]))
trainsets = [dir.stem for dir in save_resolution_dir.iterdir() if dir.is_dir()]

cells_trainsets = {}

trainset = trainsets[0]
for trainset in trainsets:
    print("Trainset: {}".format(trainset))
    
    out_trainset_dir = pl.Path("{}/{}".format(out_dir, trainset))
    
    resolutions = [dir.stem for dir in save_dir.iterdir() if dir.is_dir() and dir.stem != "mulres"]

    cells_list = []
    
    resolution = resolutions[-1]
    for resolution in resolutions:

        save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
        save_trainset_dir = pl.Path("{}/{}".format(save_resolution_dir, trainset))
    
        cells_file = pl.Path("{}/cells.parquet".format(save_trainset_dir))
        cells = pd.read_parquet(cells_file)
        
        cells = cells.sample(frac = 1 / len(resolutions),
                            random_state=seed,
                            axis = 0)
        cells = cells.sort_index()
        cells["resolution"] = resolution
        
        cells_list.append(cells)
    
    cells = pd.concat(cells_list, axis = 0)
    cells = cells.reset_index()
    cells = cells.rename({"index": "resolution_index"},
                         axis = 1)
    
    print("samples {}".format(cells.index.size))
    
    cells_out = pl.Path("{}/cells.parquet".format(out_trainset_dir))
    cells_out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(cells_out)
    
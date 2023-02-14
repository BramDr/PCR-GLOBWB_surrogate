import pathlib as pl
import pandas as pd

from utils.load_sequential_cells_reverse import load_sequential_cells_reverse

feature_dir = pl.Path("../features/saves/global_05min")
dir_out = pl.Path("./saves/global_05min")
subset_size = 2000
seed = 19920223

submasks = [dir.stem for dir in feature_dir.iterdir() if dir.is_dir()]
submasks = ["M17"]

submask = submasks[0]
for submask in submasks:
    print("Working on {}".format(submask))
    
    feature_submask_dir = pl.Path("{}/{}".format(feature_dir, submask))
    submask_out = pl.Path("{}/{}/sequential_reverse".format(dir_out, submask))
    
    cells_file = pl.Path("{}/cells.csv".format(feature_submask_dir))
    cells = pd.read_csv(cells_file, index_col=0)
    
    ldd_file = pl.Path("{}/ldd.csv".format(feature_submask_dir))
    ldd = pd.read_csv(ldd_file, index_col=0)
    
    cell_indices_sequential = load_sequential_cells_reverse(cells=cells,
                                                            ldd=ldd,
                                                            subset_size=subset_size,
                                                            verbose=1)
    
    cell_indices_sequential_lens = [len(indices) for subset_indices in cell_indices_sequential for indices in subset_indices]
    print(cell_indices_sequential_lens)

    for sequence, cell_indices_subset in enumerate(cell_indices_sequential):
        for subset, cell_indices in enumerate(cell_indices_subset):
            cells_sequence_subset = cells.loc[cell_indices]

            cells_out = pl.Path(
                "{}/sequence_{}/cells_{}.csv".format(submask_out, sequence, subset))
            cells_out.parent.mkdir(parents=True,
                                   exist_ok=True)
            cells_sequence_subset.to_csv(cells_out)

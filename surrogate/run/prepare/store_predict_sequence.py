import pathlib as pl
import datetime as dt

import pandas as pd

from surrogate.run.prepare.utils.get_area_sequence import get_area_sequence

base_dir = pl.Path("./saves")
save_dir = pl.Path("./saves/predict")
out_dir = pl.Path("./saves/predict")
time_start = dt.date(2000, 1, 1)
time_end = dt.date(2010, 12, 31)

resolutions = [dir.stem for dir in save_dir.iterdir() if dir.is_dir()]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))

    base_resolution_dir = pl.Path("{}/{}".format(base_dir, resolution))
    save_resolution_dir = pl.Path("{}/{}".format(save_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))
    
    cells_file = pl.Path("{}/cells.parquet".format(base_resolution_dir))
    cells = pd.read_parquet(cells_file)
    
    domains = cells["domain"].unique()
    
    domain = domains[0]
    for domain in domains:
        print("\tDomain: {}".format(domain))
        
        save_domain_dir = pl.Path("{}/{}".format(save_resolution_dir, domain))
        out_domain_dir = pl.Path("{}/{}".format(out_resolution_dir, domain))
        
        cells_out = pl.Path("{}/cells.parquet".format(out_domain_dir))
        if cells_out.exists():
            print("\t\tAlready exists")
            continue
    
        domain_cells = cells.loc[cells["domain"] == domain].copy()
        domain_cells = domain_cells.sort_index()
                
        ldd_file = pl.Path("{}/ldd.parquet".format(save_domain_dir))
        ldd = pd.read_parquet(ldd_file)
        
        pits_sequences = get_area_sequence(cells = domain_cells,
                                           ldd = ldd)
    
        domain_cells["pit"] = -1
        domain_cells["sequence"] = -1
        for pit_index, pit_sequences in enumerate(pits_sequences):
            for sequence_index, sequence_indices in enumerate(pit_sequences):
                domain_cells.loc[sequence_indices, "pit"] = pit_index
                domain_cells.loc[sequence_indices, "sequence"] = sequence_index
        
        domain_cells["start"] = time_start
        domain_cells["end"] = time_end
        
        domain_cells = domain_cells.astype({"pit": "int32",
                                            "sequence": "int32",
                                            "start": "object",
                                            "end": "object"})
        
        cells_out.parent.mkdir(parents=True, exist_ok=True)
        domain_cells.to_parquet(cells_out)
        
        

import pathlib as pl

import numpy as np
import pandas as pd

base_dir = pl.Path("./saves")
save_dir = pl.Path("./saves/predict")
out_dir = pl.Path("./saves/predict")

resolutions = [dir.stem for dir in save_dir.iterdir() if dir.is_dir()]

resolution = resolutions[0]
for resolution in resolutions:
    print("Resolution: {}".format(resolution))

    base_resolution_dir = pl.Path("{}/{}".format(base_dir, resolution))
    out_resolution_dir = pl.Path("{}/{}".format(out_dir, resolution))
    
    cells_file = pl.Path("{}/cells.parquet".format(base_resolution_dir))
    cells = pd.read_parquet(cells_file)
    
    ldd_file = pl.Path("{}/ldd.parquet".format(base_resolution_dir))
    ldd = pd.read_parquet(ldd_file)
    
    domains = cells["domain"].unique()
    
    domain = domains[0]
    for domain in domains:
        print("\tDomain: {}".format(domain))
        
        out_domain_dir = pl.Path("{}/{}".format(out_resolution_dir, domain))
        
        ldd_out = pl.Path("{}/ldd.parquet".format(out_domain_dir))
        if ldd_out.exists():
            print("\t\tAlready exists")
            #continue
    
        domain_cells = cells.loc[cells["domain"] == domain]

        up_sel = np.isin(ldd.index, domain_cells.index)
        down_sel = np.isin(ldd["downstream"], domain_cells.index)
        ldd_sel = np.logical_and(up_sel, down_sel)
        domain_ldd = ldd.loc[up_sel].copy()
        
        # Upstream
        domain_ldd["nupstream"] = 0
        domain_ldd["upstream"] = None
        for id in domain_ldd.index:
            domain_ldd.at[id, "upstream"] = []
            
        for id in domain_ldd.index:
            if not domain_ldd.at[id, "pit"]:
                domain_ldd.at[domain_ldd.at[id, "downstream"], "upstream"].append(id)
                domain_ldd.at[domain_ldd.at[id, "downstream"], "nupstream"] += 1
        
        # Waterbody upstream
        domain_ldd["waterbody_nupstream"] = 0
        domain_ldd["waterbody_upstream"] = None
        for wb_id in domain_ldd["waterbody_id"].unique():
            if wb_id == 0:
                continue
            ids = domain_ldd.index[domain_ldd["waterbody_id"] == wb_id]
            
            upstream_sel = np.logical_and(domain_ldd["downstream_waterbody_id"] == wb_id, domain_ldd["waterbody_transfer"])
            upstream = domain_ldd.index[upstream_sel].to_list()
            
            for id in ids:
                domain_ldd.at[id, "waterbody_upstream"] = upstream
                domain_ldd.at[id, "waterbody_nupstream"] = len(upstream)
        
        # Save
        domain_ldd = domain_ldd.astype({"nupstream": "int8",
                                        "upstream": "object",
                                        "waterbody_nupstream": "int32",
                                        "waterbody_upstream": "object"})
        
        ldd_out.parent.mkdir(parents=True,
                            exist_ok=True)
        domain_ldd.to_parquet(ldd_out)

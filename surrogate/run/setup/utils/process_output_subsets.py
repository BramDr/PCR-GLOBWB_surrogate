from typing import Sequence, Optional
import pathlib as pl


def process_output_subset(dir_out: pl.Path,
                          dataset: str,
                          feature: str,
                          combined: bool = False,
                          domain: str = ".") -> tuple[pl.Path, pl.Path, bool]:

    meta_out = pl.Path("{}/{}/{}/{}_meta.pkl".format(dir_out, dataset, domain, feature))
    
    if not combined:
        array_out = pl.Path("{}/{}/{}/{}.npy".format(dir_out, dataset, domain, feature))
    else:
        array_out = pl.Path("{}/{}/{}/{}.npz".format(dir_out, dataset, domain, feature))
        
    process_flag = True
    if meta_out.exists() and array_out.exists():
        process_flag = False
        
    return meta_out, array_out, process_flag

def process_output_subsets(out_subsets: Sequence[pl.Path],
                          dataset: str,
                          feature: str,
                          combine: bool = False,
                          domain: str = ".") -> tuple[list[pl.Path], list[pl.Path], list[bool]]:
        
        # Prepare subset data
        meta_out_subsets = []
        array_out_subsets = []
        process_flag_subsets = []
        for dir_out in out_subsets:

            meta_out, array_out, process_flag = process_output_subset(dir_out=dir_out,
                                                                     dataset=dataset,
                                                                     domain=domain,
                                                                     feature=feature,
                                                                     combined=combine)
            meta_out_subsets.append(meta_out)
            array_out_subsets.append(array_out)
            process_flag_subsets.append(process_flag)
        
        return meta_out_subsets, array_out_subsets, process_flag_subsets
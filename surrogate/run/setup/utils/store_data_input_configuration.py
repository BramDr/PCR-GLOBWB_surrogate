from typing import Optional
import configparser as cp
import pathlib as pl
import re

import numpy as np

from .store_data_value import store_data_value


def store_data_input_configuration(samples: np.ndarray,
                                   lons: np.ndarray,
                                   lats: np.ndarray,
                                   dates: np.ndarray,
                                   configuration_file: pl.Path,
                                   dir_out: pl.Path = pl.Path("."),
                                   prefix: Optional[pl.Path] = None,
                                   verbose: int = 1) -> None:

    if verbose > 0:
        print("Working on configuration file {}".format(configuration_file), flush=True)
        
    if not configuration_file.exists():
        raise ValueError("{} does not exists".format(configuration_file))

    # Load configuration
    configuration = cp.RawConfigParser()
    configuration.read(configuration_file)

    conf_prefix = configuration.get(section="globalOptions",
                                    option="inputdir")
    conf_prefix = pl.Path(conf_prefix)
        
    if prefix is not None:
        conf_prefix = "{}/{}".format(prefix, conf_prefix)
        conf_prefix = pl.Path(conf_prefix)

    for section in configuration.sections():        
        section_out = "{}/{}".format(dir_out, section)
        section_out = pl.Path(section_out)
        
        if section == "globalOptions" or section == "reportingOptions":
            continue
            
        for option in configuration.options(section):
            option_out = "{}/{}".format(section_out, option)
            option_out = pl.Path(option_out)

            if option.endswith(("ini")):
                continue
            
            levels = None
            if option == "relativeelevationfiles":
                levels = configuration.get(section=section,
                                           option="relativeelevationlevels").split(",")
                levels = ["{:04d}".format(int(float(level) * 100)) for level in levels]
                    
            value = configuration.get(section=section,
                                      option=option)        
            values = [value]
            if levels is not None:
                values = [re.sub(pattern="%04d", repl=level, string=value) for level in levels]
            
            for value in values:                
                if not value.endswith((".map", ".nc", ".nc4")):
                    continue
                
                value = pl.Path(value)
                value_out = "{}/{}".format(option_out, value.stem)
                value_out = pl.Path(value_out)
                                
                if verbose > 0:
                    print("Section {} - option {} - value {}".format(section, option, value.stem))
                
                store_data_value(value=value,
                                 samples=samples,
                                 lons=lons,
                                 lats=lats,
                                 dates=dates,
                                 dir_out=value_out,
                                 prefix=conf_prefix)

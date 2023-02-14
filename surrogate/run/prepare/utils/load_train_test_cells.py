import copy
import datetime as dt
import pandas as pd


def load_train_test_cells(cells: pd.DataFrame,
                          perc_train_spat: float,
                          perc_train_temp: float,
                          seed: int = 19920223) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    sim_time_start = dt.datetime.strptime(cells["start"].iloc[0], "%Y-%m-%d")
    sim_time_end = dt.datetime.strptime(cells["end"].iloc[0], "%Y-%m-%d")

    # Training sample
    cells_train = cells.sample(frac=perc_train_spat, random_state=seed)
    val_time_split = sim_time_start + \
        (sim_time_end - sim_time_start) * perc_train_temp
    cells_train["start"] = sim_time_start
    cells_train["end"] = val_time_split - dt.timedelta(days=1)
    cells_train = cells_train.sort_index()

    # Validation sample
    cells_val_spat = cells.drop(cells_train.index)
    cells_val_spat["start"] = sim_time_start
    cells_val_spat["end"] = val_time_split - dt.timedelta(days=1)
    cells_val_spat = cells_val_spat.sort_index()

    cells_val_temp = copy.copy(cells_train)
    cells_val_temp["start"] = val_time_split
    cells_val_temp["end"] = sim_time_end
    cells_val_temp = cells_val_temp.sort_index()

    cells_val_spattemp = copy.copy(cells_val_spat)
    cells_val_spattemp["start"] = val_time_split
    cells_val_spattemp["end"] = sim_time_end
    cells_val_spattemp = cells_val_spattemp.sort_index()

    return cells_train, cells_val_spat, cells_val_temp, cells_val_spattemp

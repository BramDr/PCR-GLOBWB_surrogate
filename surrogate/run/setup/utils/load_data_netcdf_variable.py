import netCDF4 as nc
import numpy as np

from .get_data_indices import indices_from_map_array
from .get_data_indices import indices_from_map_array_flat


def load_data_netcdf_variable_static_flat(variable: nc.Variable,
                                          s_indices: np.ndarray) -> np.ndarray:

    values = variable[:, :]
    values = indices_from_map_array_flat(map=values,
                                         s_indices=s_indices)

    return values


def load_data_netcdf_variable_temporal_flat(variable: nc.Variable,
                                            s_indices: np.ndarray,
                                            d_indices: np.ndarray) -> np.ndarray:

    temporal_values = []
    for index in d_indices:
        map_values = variable[index, :, :]
        map_values = indices_from_map_array_flat(map=map_values,
                                                 s_indices=s_indices)
        temporal_values.append(map_values)

    values = np.stack(temporal_values, axis=-1)
    return values


def load_data_netcdf_variable_static(variable: nc.Variable,
                                      x_indices: np.ndarray,
                                      y_indices: np.ndarray) -> np.ndarray:

    values = variable[:, :]
    values = indices_from_map_array(map=values,
                                    x_indices=x_indices,
                                    y_indices=y_indices)

    return values


def load_data_netcdf_variable_temporal(variable: nc.Variable,
                                        x_indices: np.ndarray,
                                        y_indices: np.ndarray,
                                        d_indices: np.ndarray) -> np.ndarray:

    temporal_values = []
    for index in d_indices:
        map_values = variable[index, :, :]
        map_values = indices_from_map_array(map=map_values,
                                            x_indices=x_indices,
                                            y_indices=y_indices)
        temporal_values.append(map_values)

    values = np.stack(temporal_values, axis=-1)
    return values

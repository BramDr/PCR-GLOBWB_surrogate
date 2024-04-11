import pathlib as pl
import subprocess as sp


def get_map_attributes(map_file: pl.Path) -> dict:

    cOut, _ = sp.Popen("mapattr -p {}".format(map_file),
                       stdout=sp.PIPE,
                       stderr=sp.PIPE,
                       shell=True).communicate()

    resolution = float(cOut.split()[7])
    resolution = round(resolution * 360000.)/360000.
    x_len = int(cOut.split()[5])
    y_len = int(cOut.split()[3])
    x_corner_value = float(cOut.split()[17])
    y_corner_value = float(cOut.split()[19])
    attributes = {"resolution": float(resolution),
                  "x_len": x_len,
                  "y_len":  y_len,
                  "x_corner_value": x_corner_value,
                  "y_corner_value": y_corner_value}

    return attributes

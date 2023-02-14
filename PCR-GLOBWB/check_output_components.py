from pathlib import Path
import re
from netCDF4 import Dataset

output_dir = Path("./output/RhineMeuse/netcdf/")
output_files = [file for file in output_dir.iterdir() if not file.is_dir()]
output_names = [re.sub(pattern="_dailyTot_.*", repl="", string=str(file.name))
                for file in output_files]

storage_combinations = {"totalWaterStorageThickness": ["totalActiveStorageThickness", "storGroundwaterFossil"],
                        "totalActiveStorageThickness": ["interceptStor", "snowCoverSWE", "snowFreeWater", "topWaterLayer", "storUppTotal", "storLowTotal", "surfaceWaterStorage", "storGroundwater"]}
evaporation_combinations = {"totalEvaporation": ["actualET", "waterBodyActEvaporation"],
                            "actualET": ["interceptEvap", "actSnowFreeWaterEvap", "topWaterLayerEvap", "actBareSoilEvap", "actTranspiTotal"],
                            "actTranspiTotal": ["actTranspiUppTotal", "actTranspiLowTotal"]}
runoff_combinations = {"totalRunoff": ["local_water_body_flux", "runoff"],
                       "runoff": ["directRunoff", "interflowTotal", "baseflow"],
                       "local_water_body_flux": ["-surfaceWaterInf", "-surfaceWaterAbstraction", "nonIrrReturnFlow", "-waterBodyActEvaporation"]}
combinations = {**storage_combinations, **evaporation_combinations, **runoff_combinations}

total_name, comp_names = next(iter(combinations.items()))
for total_name, comp_names in combinations.items():
    total_index = output_names.index(total_name)
    total_file = output_files[total_index]
    total_dataset = Dataset(total_file)
    total_variable = [variable for variable in total_dataset.variables.values(
    ) if variable.name not in ['time', 'lon', 'lat']][0]
    total = total_variable[:, :, :]

    comp_sum = total * 0.0
    for comp_name in comp_names:
        comp_sign = comp_name[0]
        comp_neg = comp_sign == "-"
        if comp_neg:
            comp_name = comp_name[1:]

        comp_index = output_names.index(comp_name)
        comp_file = output_files[comp_index]
        comp_dataset = Dataset(comp_file)
        comp_variable = [variable for variable in comp_dataset.variables.values(
        ) if variable.name not in ['time', 'lon', 'lat']][0]
        comp = comp_variable[:, :, :]

        if not comp_neg:
            comp_sum += comp
        else:
            comp_sum -= comp

    diff = total - comp_sum    
    print("{}: {}".format(total_name, comp_names))
    print("difference: {: .2e} to {: .2e}".format(diff.min(), diff.max()))
